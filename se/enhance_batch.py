#!/usr/bin/env python3
"""SE 批量增强（板端）：对 benchmark SE 子集的 noisy 逐条跑增强模型 -> enhanced_dir。
每条调用模型仓库自带 demo（命令见 tools/model_registry.py），记录端到端 RTF。
用法: python enhance_batch.py --model gtcrn|fastenhancer [--chip ax650] [--out_dir DIR]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, remap
from model_registry import REGISTRY


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[k for k, v in REGISTRY.items() if v["kind"] == "se"])
    ap.add_argument("--chip", default=cfg["models"]["chip"])
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    model_dir = Path(cfg["paths"]["model_root"]) / cfg["models"]["se"][args.model]["dir"]
    scp = Path(cfg["paths"]["data_root"]) / "benchmark" / "se" / "voicebank_demand" / "subset.scp"
    if not scp.exists():
        print(f"无子集 {scp}"); return
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["paths"]["data_root"]) / "benchmark" / "se" / "enhanced" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [l.split("\t") for l in scp.read_text().splitlines() if l.strip()]
    lim = int(os.environ.get("LIMIT", "0"))
    if lim > 0:
        rows = rows[:lim]
    infer_s = audio_s = 0.0
    ok = 0
    for uid, clean, noisy in rows:
        noisy = remap(noisy)
        outp = out_dir / f"{uid}.wav"
        workdir, argv = spec["builder"](model_dir, args.chip, noisy, outp.resolve())
        t0 = time.perf_counter()
        r = subprocess.run(argv, cwd=workdir, capture_output=True, text=True)
        infer_s += time.perf_counter() - t0
        try:
            audio_s += sf.info(noisy).duration
        except Exception:
            pass
        if outp.exists():
            ok += 1
        elif ok == 0 and r.returncode != 0:  # 首条失败即打印诊断
            print(f"  [{args.model}] 首条失败: {' '.join(argv[:4])}...\n  stderr: {r.stderr[-400:]}")
    rtf = infer_s / audio_s if audio_s else float("nan")
    print(f">>> {args.model}: 增强 {ok}/{len(rows)} 条 -> {out_dir}  RTF={rtf:.4f}")
    # RTF 记到旁路文件供 run.sh 读取
    (out_dir / "_rtf.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
