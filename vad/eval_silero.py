#!/usr/bin/env python3
"""SileroVAD 在 benchmark 子集上评测（板端）。
  - LibriVAD/AISHELL/TEN VAD 逐句集：流式 chunk→每帧概率，与采样级标签(降到32ms帧)比，算 F1/AUC
  - Picovoice 长流：整条流式推理，与 benchmark_labels.txt(32ms帧,1=未知忽略)比
输出行追加到 results/vad.csv。
用法: python eval_silero.py --backend ax650 [--dataset librivad|aishell1|ten_official|picovoice]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, CmmDelta, read_rss_mb, remap
import metrics_vad as V


def _frame_probs(model, wav, sr=16000, win=512):
    """流式 chunk 推理：每 512 采样(32ms)出一个语音概率。
    避开 vendor audio_forward 的 np.pad bug，且更贴近真实流式 VAD。"""
    probs = []
    for i in range(0, len(wav) - win + 1, win):
        p = model(wav[i:i + win], sr)
        probs.append(float(p.item() if hasattr(p, "item") else p))
    model.reset_states()
    return np.asarray(probs)


def eval_uttlist(model, scp, sr=16000, frame_ms=32):
    """逐句集：返回 (aggregate指标, rtf)。"""
    import os
    items = []
    infer_s = audio_s = 0.0
    lines = [l for l in Path(scp).read_text().splitlines() if l.strip()]
    lim = int(os.environ.get("LIMIT", "0"))
    if lim > 0:
        lines = lines[:lim]
    for line in lines:
        uid, wav_path, lab_path = line.split("\t")[:3]
        wav_path, lab_path = remap(wav_path), remap(lab_path)
        from silero_vad_axera import read_audio
        wav = read_audio(wav_path, sampling_rate=sr)
        t0 = time.perf_counter()
        probs = _frame_probs(model, wav, sr)
        infer_s += time.perf_counter() - t0
        audio_s += len(wav) / sr
        ref_samples = np.load(lab_path)
        ref_frames = V.samples_to_frames(ref_samples, sr, frame_ms)
        n = min(len(probs), len(ref_frames))
        if n == 0:
            continue
        items.append((probs[:n], ref_frames[:n]))
    res = V.aggregate(items, ignore_label=None, threshold=0.5, frame_ms=frame_ms)
    res["rtf"] = infer_s / audio_s if audio_s else float("nan")
    return res


def eval_picovoice(model, info_txt, sr=16000):
    import os
    import soundfile as sf
    kv = dict(l.split("=", 1) for l in Path(info_txt).read_text().splitlines() if "=" in l)
    ref = np.loadtxt(remap(kv["labels"].strip()), dtype=int)  # 0=sil 1=unknown 2=speech
    # 长流≈20h/2.2GB，只读前 PICO_SEC 秒（默认 600s），避免全量载入 OOM
    cap = int(os.environ.get("PICO_SEC", "600"))
    n_frames = cap * sr if cap > 0 else -1
    wav, _sr = sf.read(remap(kv["audio"].strip()), frames=n_frames, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if cap > 0:
        ref = ref[:int(cap * 1000 / 32)]
    t0 = time.perf_counter()
    probs = _frame_probs(model, wav, sr)
    rtf = (time.perf_counter() - t0) / (len(wav) / sr)
    n = min(len(probs), len(ref))
    res = V.aggregate([(probs[:n], ref[:n])], ignore_label=1, threshold=0.5)
    res["rtf"] = rtf
    return res


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ax650")
    ap.add_argument("--dataset", nargs="+",
                    default=["librivad", "aishell1", "ten_official", "picovoice"])
    args = ap.parse_args()
    dr = Path(cfg["paths"]["data_root"]) / "benchmark" / "vad"
    csv = REPO / "results" / "vad.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        csv.write_text("module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note\n")

    # CMM/RSS 都在「载入模型后、推理前」定格：
    #  - CMM 基线须在 load 之前取（axengine 建 handle 时才分配 NPU 内存），之后一次采样即可
    #    ——CMM 分配完就不变，不需要后台轮询；轮询线程抢 GIL 会让 RTF 虚高约 5%
    #  - RSS 同样只取此刻，否则会量进本脚本为算 AUC 累积的 probs/labels 数组（实测 45→238MB），
    #    那不是模型占用
    cmm_probe = CmmDelta()
    from silero_vad_axera import load_silero_vad
    model = load_silero_vad(args.backend)
    cmm_probe.sample()
    cmm, osm = cmm_probe.delta, read_rss_mb()

    def rec(ds, res):
        lang = "zh" if ds == "aishell1" else "en"
        with open(csv, "a") as f:
            for m in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
                f.write(f"vad,silero,{ds},{lang},{m},{res[m]:.4f},{res.get('rtf',''):.4f},"
                        f"{'' if cmm is None else f'{cmm:.1f}'},{osm or ''},board frame=32ms\n")
        print(f"  [{ds}] F1={res['f1']:.3f} AUC={res['roc_auc']:.3f} "
              f"acc={res['accuracy']:.3f} RTF={res.get('rtf',float('nan')):.4f} "
              f"CMM={'—' if cmm is None else f'{cmm:.1f}'}MB")

    for ds in args.dataset:
        if ds == "picovoice":
            info = dr / "picovoice" / "info.txt"
            if info.exists():
                rec("picovoice", eval_picovoice(model, info))
        else:
            scp = dr / ds / "subset.scp"
            if scp.exists():
                rec(ds, eval_uttlist(model, scp))
    print(f">>> VAD 评测完成 -> {csv}")


if __name__ == "__main__":
    main()
