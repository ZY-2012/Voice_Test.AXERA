#!/usr/bin/env python3
"""SE 批量增强（板端，载入一次 + warmup + 热启动 RTF）。
  - 模型只加载一次，进程内循环推理（避免 per-file 冷启动）
  - RTF = 纯推理/前后端耗时 / 音频时长，跳过首条(warmup)，不含模型加载
  - gtcrn: 复用 demo 的 init_caches/update_caches + STFT/帧循环/ISTFT
  - fastenhancer: FastEnhancer SDK（load once, enhance per file）
输出: enhanced/<id>.wav + _rtf.txt(热启动RTF)
用法: python enhance_warm.py --model gtcrn|fastenhancer [--chip ax650] [--limit N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, remap

_GTCRN_AXM = {"ax650": "gtcrn_650.axmodel", "ax630c": "gtcrn_630.axmodel", "ax620q": "gtcrn_615.axmodel"}


def _load_scp(cfg, limit):
    scp = Path(cfg["paths"]["data_root"]) / "benchmark" / "se" / "voicebank_demand" / "subset.scp"
    rows = []
    for line in scp.read_text().splitlines():
        if not line.strip():
            continue
        uid, clean, noisy = line.split("\t")[:3]
        rows.append((uid, remap(noisy)))
    if limit > 0:
        rows = rows[:limit]
    return rows


def run_gtcrn(model_dir, chip, rows, out_dir):
    import torch
    import librosa
    from axengine import InferenceSession
    sys.path.insert(0, str(model_dir))
    from demo_gtcrn_ax import init_caches, update_caches
    n_fft, hop, sr = 512, 256, 16000
    session = InferenceSession(str(model_dir / "models" / _GTCRN_AXM.get(chip, "gtcrn_650.axmodel")))  # load once
    win = torch.hann_window(n_fft).pow(0.5)
    iwin = torch.from_numpy(np.hanning(n_fft) ** 0.5).float()
    infer_s = audio_s = 0.0
    ok = 0
    for k, (uid, noisy) in enumerate(rows):
        audio, s = sf.read(noisy, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        if s != sr:
            audio = librosa.resample(audio, orig_sr=s, target_sr=sr)
        t0 = time.perf_counter()
        spec = torch.stft(torch.from_numpy(audio), n_fft, hop, n_fft, win, return_complex=False).numpy()[None, ...]
        cache = init_caches()
        frames = []
        for i in range(spec.shape[2]):
            inp = {"mix": spec[:, :, i:i + 1, :]}
            inp.update(cache)
            out = session.run(None, inp)
            frames.append(out[0])
            cache = update_caches(cache, out)
        es = np.concatenate(frames, axis=2)
        cplx = es[0, :, :, 0] + 1j * es[0, :, :, 1]
        real = torch.from_numpy(cplx.real).unsqueeze(0).contiguous()
        imag = torch.from_numpy(cplx.imag).unsqueeze(0).contiguous()
        wav = torch.istft(torch.complex(real, imag), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=iwin, center=True).squeeze(0).numpy().astype(np.float32)
        dt = time.perf_counter() - t0
        if k > 0:  # 跳过首条 warmup
            infer_s += dt
            audio_s += len(audio) / sr
        sf.write(out_dir / f"{uid}.wav", wav, sr)
        ok += 1
    return ok, (infer_s / audio_s if audio_s else float("nan"))


def run_fastenhancer(model_dir, chip, rows, out_dir):
    sys.path.insert(0, str(model_dir / "python"))
    from fastenhancer_sdk import FastEnhancer
    fe = FastEnhancer(str(model_dir / "models" / "16k" / "model.axmodel"))  # load once
    infer_s = audio_s = 0.0
    ok = 0
    for k, (uid, noisy) in enumerate(rows):
        wav, s = sf.read(noisy, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if s != fe.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=s, target_sr=fe.sr)
        t0 = time.perf_counter()
        out = fe.enhance(wav)
        dt = time.perf_counter() - t0
        if k > 0:
            infer_s += dt
            audio_s += len(wav) / fe.sr
        sf.write(out_dir / f"{uid}.wav", out, fe.sr)
        ok += 1
    return ok, (infer_s / audio_s if audio_s else float("nan"))


def run_deepfilternet3(model_dir, chip, rows, out_dir):
    """DeepFilterNet3：48kHz 模型。载入一次，noisy 16k→48k 重采样增强，输出降回 16k（与 clean 同域评估）。"""
    import librosa
    sys.path.insert(0, str(model_dir))
    from deepfilternet3_ax import DeepFilterNet3
    enh = DeepFilterNet3(model_dir / "axmodels")   # load once
    infer_s = audio_s = 0.0
    ok = 0
    for k, (uid, noisy) in enumerate(rows):
        wav, s = sf.read(noisy, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if s != 48000:
            wav = librosa.resample(wav, orig_sr=s, target_sr=48000)
        t0 = time.perf_counter()
        out = enh.enhance(wav)
        dt = time.perf_counter() - t0
        if k > 0:  # warmup 跳过首条
            infer_s += dt
            audio_s += len(wav) / 48000
        out16 = librosa.resample(out, orig_sr=48000, target_sr=16000)
        sf.write(out_dir / f"{uid}.wav", out16, 16000)
        ok += 1
    return ok, (infer_s / audio_s if audio_s else float("nan"))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gtcrn", "fastenhancer", "deepfilternet3"])
    ap.add_argument("--chip", default=cfg["models"]["chip"])
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    args = ap.parse_args()
    model_dir = Path(cfg["paths"]["model_root"]) / cfg["models"]["se"][args.model]["dir"]
    out_dir = Path(cfg["paths"]["data_root"]) / "benchmark" / "se" / "enhanced" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_scp(cfg, args.limit)
    fn = {"gtcrn": run_gtcrn, "fastenhancer": run_fastenhancer,
          "deepfilternet3": run_deepfilternet3}[args.model]
    ok, rtf = fn(model_dir, args.chip, rows, out_dir)
    print(f">>> {args.model}: 增强 {ok} 条 -> {out_dir}  热启动RTF={rtf:.4f}")
    (out_dir / "_rtf.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
