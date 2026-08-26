#!/usr/bin/env python3
"""TTS 指标：
  - cer_loopback: 合成音频经已适配 ASR 识别 → 与输入文本算 CER/WER（可懂度，越低越好）
  - mcd:          梅尔倒谱距离（有参考音频时，越低越好）
  - rtf:          合成实时率（由合成端记录，这里仅聚合）

cer_loopback 复用 tools/metrics_asr.py 的 CER/WER 口径。
MCD 用 DTW 对齐后的 MFCC 距离（librosa）。
"""
import argparse
from pathlib import Path

import numpy as np

from metrics_asr import score as asr_score, load_text


def mcd(ref_wav, syn_wav, sr=16000, n_mfcc=13):
    """Mel-Cepstral Distortion（dB），DTW 对齐。需要 librosa。"""
    import librosa
    r, _ = librosa.load(ref_wav, sr=sr)
    s, _ = librosa.load(syn_wav, sr=sr)
    mr = librosa.feature.mfcc(y=r, sr=sr, n_mfcc=n_mfcc).T[:, 1:]  # 去能量维
    ms = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=n_mfcc).T[:, 1:]
    D, wp = librosa.sequence.dtw(mr.T, ms.T, metric="euclidean")
    k = 10.0 / np.log(10) * np.sqrt(2)
    dists = [np.sqrt(((mr[i] - ms[j]) ** 2).sum()) for i, j in wp]
    return float(k * np.mean(dists))


def loopback_cer(text_ref_path, asr_hyp_path, lang="zh"):
    """text_ref_path: TTS 输入文本(utt_id\t文本)；asr_hyp_path: ASR 识别合成音后的结果。"""
    ref = load_text(text_ref_path)
    hyp = load_text(asr_hyp_path)
    return asr_score(hyp, ref, lang=lang)


def mcd_over_dir(ref_dir, syn_dir, sr=16000):
    ref_dir, syn_dir = Path(ref_dir), Path(syn_dir)
    vals = []
    for p in sorted(ref_dir.glob("*.wav")):
        sp = syn_dir / p.name
        if sp.exists():
            try:
                vals.append(mcd(str(p), str(sp), sr))
            except Exception:
                pass
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cer", help="回环可懂度 CER/WER")
    a.add_argument("--text_ref", required=True)
    a.add_argument("--asr_hyp", required=True)
    a.add_argument("--lang", default="zh", choices=["zh", "en"])
    b = sub.add_parser("mcd", help="梅尔倒谱距离")
    b.add_argument("--ref_dir", required=True)
    b.add_argument("--syn_dir", required=True)
    b.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()
    if args.cmd == "cer":
        r = loopback_cer(args.text_ref, args.asr_hyp, args.lang)
        m = r["metric"]
        print(f"loopback {m.upper()}: {r[m]*100:.2f}%  ({r['num_utt']} utts)")
    else:
        v, n = mcd_over_dir(args.ref_dir, args.syn_dir, args.sr)
        print(f"MCD: {v:.3f} dB  ({n} pairs)")
