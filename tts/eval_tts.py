#!/usr/bin/env python3
"""TTS 评测（板端）：
  - MCD：合成音 syn_dir/<id>.wav vs 参考音（wav.scp），DTW 对齐梅尔倒谱距离
  - 回环 CER：合成音经已适配 ASR 识别得到 hyp（--asr_hyp: id\thyp），与输入文本算 CER/WER
输出行追加到 results/tts.csv。
用法:
  python eval_tts.py --dataset aishell3 --syn_dir DIR [--asr_hyp hyp.txt]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config
import metrics_tts as T
from metrics_asr import score as asr_score, load_text


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ljspeech", "aishell3"])
    ap.add_argument("--syn_dir", required=True, help="TTS 合成音频目录 <id>.wav")
    ap.add_argument("--asr_hyp", default="", help="合成音经 ASR 识别的结果 id\\thyp")
    ap.add_argument("--model_name", default="tts-model")
    args = ap.parse_args()
    lang = "zh" if args.dataset == "aishell3" else "en"
    bench = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset
    text_ref = bench / "text"
    wav_scp = bench / "wav.scp"
    if not text_ref.exists():
        print(f"无子集 {text_ref}"); return

    csv = REPO / "results" / "tts.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        csv.write_text("module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note\n")
    out = []

    # 回环 CER
    if args.asr_hyp and Path(args.asr_hyp).exists():
        r = asr_score(load_text(args.asr_hyp), load_text(text_ref), lang=lang)
        m = r["metric"]
        out.append((f"cer_loopback_{m}", r[m], f"ASR回环({cfg['eval']['tts']['asr_model']})"))
        print(f"  回环 {m.upper()}={r[m]*100:.2f}%")

    # MCD（需参考音）
    ref_map = load_text(wav_scp) if wav_scp.exists() else {}
    import soundfile as sf, tempfile, os
    vals = []
    for uid, wav in ref_map.items():
        syn = Path(args.syn_dir) / f"{uid}.wav"
        if syn.exists():
            try:
                vals.append(T.mcd(wav[0] if isinstance(wav, list) else wav, str(syn)))
            except Exception:
                pass
    if vals:
        out.append(("mcd", float(np.mean(vals)), f"{len(vals)}对"))
        print(f"  MCD={np.mean(vals):.3f} dB ({len(vals)}对)")

    with open(csv, "a") as f:
        for metric, val, note in out:
            f.write(f"tts,{args.model_name},{args.dataset},{lang},{metric},{val:.4f},,,,{note}\n")
    print(f">>> TTS 评测完成 -> {csv}")


if __name__ == "__main__":
    main()
