#!/usr/bin/env python3
"""Pocket-TTS 中英双语（HY-2012/pocket-tts-zh-en.AXERA）载入一次批量合成（板端）。

模型目录布局（HF 推理包）：
    <model_dir>/models/{flow_step_int8.onnx, flow_ar_step_int8.onnx,
        flow_net_step_fp32.axmodel, mimi_conv_step.axmodel,
        mimi_split/mimi_transformer_step_int8.onnx, encoder/step_encoder_40f.axmodel,
        chn_jpn_yue_eng_ko_spectok.bpe.model, Vivian.wav}
    <model_dir>/board/pocket_tts_axera.py

零样本：参考音色固定用包内 Vivian.wav；同一进程内音色前缀缓存复用。

用法: python pocket_tts_zh_en_batch.py --dataset aishell3 [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, TTS_DATASETS, tts_lang  # noqa: E402

CFG = load_config()
MODEL_DIR = Path(CFG["paths"]["model_root"]) / CFG["models"]["tts"]["pocket_tts_zh_en"]["dir"]
sys.path.insert(0, str(MODEL_DIR / "board"))

from pocket_tts_axera import (  # noqa: E402
    SR, PocketTTS, TextFrontend, load_reference, write_wav,
)

MODELS = MODEL_DIR / "models"


def build_runtime(threads=4, prefill_threads=8):
    return PocketTTS(
        str(MODELS), str(MODELS), cpu_model_dir=str(MODELS),
        mimi_split_dir=str(MODELS / "mimi_split"),
        threads=threads, flow_window=512,
        npu_flow_net=True, npu_mimi_conv=True,
        flow_net_model="flow_net_step_fp32.axmodel",
        mimi_conv_model="mimi_conv_step.axmodel",
        flow_ar_model="flow_ar_step_int8.onnx",
        flow_prefill_model="flow_step_int8.onnx",
        mimi_tf_model="mimi_transformer_step_int8.onnx",
        encoder_dir=str(MODELS / "encoder"),
        prefill_threads=prefill_threads,
    )


def synthesize(tts, frontend, ref, text, pause_ms=120):
    pause = np.zeros(int(SR * pause_ms / 1000), dtype=np.float32) if pause_ms > 0 else None
    chunks = frontend.chunk_text(text)
    pieces = []
    for i, chunk in enumerate(chunks):
        ids = frontend.text2ids(chunk)
        frames = [f for f in tts.stream(ids, ref, temp=0.0, max_frames=375, seed=0)]
        pieces.append(np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32))
        if pause is not None and i < len(chunks) - 1:
            pieces.append(pause)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TTS_DATASETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefill-threads", type=int, default=8)
    ap.add_argument("--pause-ms", type=int, default=120)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    text_f = Path(CFG["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset / "text"
    rows = [l.split("\t", 1) for l in text_f.read_text(encoding="utf-8").splitlines() if "\t" in l]
    if args.limit > 0:
        rows = rows[:args.limit]
    out_dir = (Path(CFG["paths"]["data_root"]) / "benchmark" / "tts" / "syn"
               / f"pocket_tts_zh_en_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    tts = build_runtime(args.threads, args.prefill_threads)
    frontend = TextFrontend(str(MODELS / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
    ref = load_reference(str(MODELS / "Vivian.wav"))
    print(f"Pocket-TTS zh-en: 载入完成 {time.perf_counter() - t_load:.1f}s, "
          f"{len(rows)} 条待合成（参考音色 Vivian.wav）")

    infer_s = audio_s = 0.0
    ok = n_new = 0
    for k, (uid, text) in enumerate(rows):
        outp = out_dir / f"{uid}.wav"
        if not args.fresh and outp.exists() and outp.stat().st_size > 0:
            ok += 1
            continue
        t0 = time.perf_counter()
        wav = synthesize(tts, frontend, ref, text, pause_ms=args.pause_ms)
        dt = time.perf_counter() - t0
        if k > 0:
            infer_s += dt
            audio_s += len(wav) / SR
        write_wav(str(outp), wav)
        ok += 1
        n_new += 1
        if n_new % 10 == 0:
            print(f"  进度 {ok}/{len(rows)}（本次新合成 {n_new} 条，"
                  f"累计热启动 RTF {(infer_s / audio_s if audio_s else float('nan')):.4f}）")

    rtf = infer_s / audio_s if audio_s else float("nan")
    print(f">>> pocket_tts_zh_en/{args.dataset}: 合成 {ok}/{len(rows)} 条 -> {out_dir}  "
          f"热启动RTF(不含加载)={rtf:.4f}")
    if n_new > 0:
        (out_dir / "_rtf_pure.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
