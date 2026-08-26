#!/usr/bin/env python3
"""SE 评测（板端）：对 benchmark 子集算 PESQ/STOI/SI-SNR。
  - 增强路径：--enhanced_dir 指向 SE 模型输出（enhanced/<id>.wav），vs clean
  - 基线路径：无增强目录时，算 noisy vs clean 基线（验证闭环+参照增益）
输出行追加到 results/se.csv。
用法: python eval_se.py [--enhanced_dir DIR] [--model_name gtcrn]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, remap
import metrics_se as SE


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced_dir", default="")
    ap.add_argument("--model_name", default="noisy-baseline")
    ap.add_argument("--rtf", default="")
    args = ap.parse_args()
    scp = Path(cfg["paths"]["data_root"]) / "benchmark" / "se" / "voicebank_demand" / "subset.scp"
    if not scp.exists():
        print(f"无子集 {scp}，先跑 prepare_datasets.sh se"); return
    metrics = cfg["eval"]["se"]["metrics"]
    acc = {m: [] for m in metrics}
    n = 0
    for line in scp.read_text().splitlines():
        uid, clean, noisy = line.split("\t")[:3]
        clean, noisy = remap(clean), remap(noisy)
        test = str(Path(args.enhanced_dir) / f"{uid}.wav") if args.enhanced_dir else noisy
        if not Path(test).exists():
            continue
        c, sr = sf.read(clean)
        t, _ = sf.read(test)
        if c.ndim > 1: c = c.mean(1)
        if t.ndim > 1: t = t.mean(1)
        r = SE.pair_metrics(c, t, sr, metrics)
        for m in metrics:
            if not np.isnan(r[m]): acc[m].append(r[m])
        n += 1
    csv = REPO / "results" / "se.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        csv.write_text("module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note\n")
    note = "enhanced" if args.enhanced_dir else "noisy-baseline(未增强参照)"
    with open(csv, "a") as f:
        for m in metrics:
            v = float(np.mean(acc[m])) if acc[m] else float("nan")
            f.write(f"se,{args.model_name},voicebank_demand,en,{m},{v:.4f},{args.rtf},,,{note}\n")
            print(f"  {m.upper()}={v:.4f}")
    print(f">>> SE 评测完成（{n} 对，{note}）-> {csv}")


if __name__ == "__main__":
    main()
