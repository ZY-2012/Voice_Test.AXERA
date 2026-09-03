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
    """Mel-Cepstral Distortion（dB），DTW 对齐。需要 librosa。

    量纲说明（易错）：标准 MCD = (10/ln10)·√2·‖c_ref − c_syn‖，其中 c 为**自然对数**倒谱。
    librosa.feature.mfcc 内部用 power_to_db（10·log10），即已含 10/ln10 因子：
        librosa_mfcc = (10/ln10) · 自然对数倒谱
    故用 librosa MFCC 时正确系数为 **√2**，若再乘 10/ln10 会放大约 4.34 倍。

    ⚠️ 即便量纲正确，本实现与文献 MCD（SPTK/WORLD mgc，n_mels 24~40）仍不可直接比较：
    librosa 默认 n_mels=128 且 DCT norm='ortho'。且 MCD **仅同说话人有意义** ——
    预设音色模型（kokoro/melotts）与参考说话人不同，跨说话人数值无解释力。
    故默认不写入 CSV，仅作同说话人零样本克隆的辅助诊断。
    """
    import librosa
    r, _ = librosa.load(ref_wav, sr=sr)
    s, _ = librosa.load(syn_wav, sr=sr)
    mr = librosa.feature.mfcc(y=r, sr=sr, n_mfcc=n_mfcc).T[:, 1:]  # 去能量维
    ms = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=n_mfcc).T[:, 1:]
    D, wp = librosa.sequence.dtw(mr.T, ms.T, metric="euclidean")
    k = np.sqrt(2)   # librosa MFCC 已是 dB 量纲，不再乘 10/ln10
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
