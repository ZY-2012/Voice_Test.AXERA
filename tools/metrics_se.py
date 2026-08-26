#!/usr/bin/env python3
"""SE（语音增强）指标：PESQ / STOI / SI-SNR。
参考 clean 与增强后 enhanced（或带噪 noisy）逐对计算。

用法:
  python metrics_se.py --clean_dir clean/ --test_dir enhanced/   # 增强后 vs 干净
  # 也可 --test_dir noisy/ 得到未处理基线，用于对照增强增益
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def si_snr(ref, est, eps=1e-8):
    """Scale-Invariant SNR（dB）。ref/est 等长 1D。"""
    ref = ref - ref.mean()
    est = est - est.mean()
    t = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    noise = est - t
    return float(10 * np.log10((np.dot(t, t) + eps) / (np.dot(noise, noise) + eps)))


def _align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def pair_metrics(clean, test, sr, want):
    clean, test = _align(clean, test)
    out = {}
    if "si_snr" in want:
        out["si_snr"] = si_snr(clean, test)
    if "pesq" in want:
        try:
            from pesq import pesq
            mode = "wb" if sr >= 16000 else "nb"
            out["pesq"] = float(pesq(16000 if sr >= 16000 else 8000, clean, test, mode))
        except Exception as e:
            out["pesq"] = float("nan")
    if "stoi" in want:
        try:
            from pystoi import stoi
            out["stoi"] = float(stoi(clean, test, sr, extended=False))
        except Exception:
            out["stoi"] = float("nan")
    return out


def evaluate(clean_dir, test_dir, metrics=("pesq", "stoi", "si_snr")):
    clean_dir, test_dir = Path(clean_dir), Path(test_dir)
    ids = sorted(p.stem for p in clean_dir.glob("*.wav"))
    acc = {m: [] for m in metrics}
    n = 0
    for uid in ids:
        tf = test_dir / f"{uid}.wav"
        if not tf.exists():
            continue
        c, sr = sf.read(clean_dir / f"{uid}.wav")
        t, _ = sf.read(tf)
        if c.ndim > 1:
            c = c.mean(1)
        if t.ndim > 1:
            t = t.mean(1)
        r = pair_metrics(c, t, sr, metrics)
        for m in metrics:
            if m in r and not np.isnan(r[m]):
                acc[m].append(r[m])
        n += 1
    summary = {m: (float(np.mean(v)) if v else float("nan")) for m, v in acc.items()}
    summary["num_pairs"] = n
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--test_dir", required=True, help="增强后音频目录（或 noisy 基线）")
    ap.add_argument("--metrics", nargs="+", default=["pesq", "stoi", "si_snr"])
    args = ap.parse_args()
    s = evaluate(args.clean_dir, args.test_dir, args.metrics)
    print(f"pairs={s['num_pairs']}  " +
          "  ".join(f"{m.upper()}={s[m]:.4f}" for m in args.metrics))
