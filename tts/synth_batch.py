#!/usr/bin/env python3
"""TTS 批量合成（板端）：把 benchmark TTS 子集的 text 逐句合成 -> syn_dir/<id>.wav。
每句调用模型仓库自带 demo/binary（命令见 tools/model_registry.py），记录端到端 RTF。
用法: python synth_batch.py --model kokoro|melotts|zipvoice|cosyvoice2 --dataset aishell3|ljspeech [--chip ax650]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config
from model_registry import REGISTRY


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[k for k, v in REGISTRY.items() if v["kind"] == "tts"])
    ap.add_argument("--dataset", required=True, choices=["aishell3", "ljspeech"])
    ap.add_argument("--chip", default=cfg["models"]["chip"])
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只合成前 N 条（重模型如 cosyvoice2 用）")
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    lang = "zh" if args.dataset == "aishell3" else "en"
    if lang not in spec.get("langs", ["zh", "en"]):
        print(f"  {args.model} 不支持 {lang}，跳过"); return
    model_dir = Path(cfg["paths"]["model_root"]) / cfg["models"]["tts"][args.model]["dir"]
    text_f = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset / "text"
    if not text_f.exists():
        print(f"无子集 {text_f}"); return
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / "syn" / f"{args.model}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [l.split("\t", 1) for l in text_f.read_text(encoding="utf-8").splitlines() if "\t" in l]
    if args.limit > 0:
        rows = rows[:args.limit]
    infer_s = audio_s = 0.0
    ok = 0
    for uid, text in rows:
        outp = out_dir / f"{uid}.wav"
        workdir, argv = spec["builder"](model_dir, args.chip, lang, text, outp.resolve())
        t0 = time.perf_counter()
        r = subprocess.run(argv, cwd=workdir, capture_output=True, text=True)
        infer_s += time.perf_counter() - t0
        # cosyvoice2 输出 output*.wav，需改名到 outp
        if spec.get("out_glob") and not outp.exists():
            cand = sorted(Path(workdir).glob(spec["out_glob"]))
            if cand:
                shutil.move(str(cand[-1]), outp)
        if outp.exists():
            try:
                audio_s += sf.info(outp).duration
            except Exception:
                pass
            ok += 1
        elif ok == 0 and r.returncode != 0:
            print(f"  [{args.model}] 首条失败: {' '.join(str(a) for a in argv[:4])}...\n  stderr: {r.stderr[-400:]}")
    rtf = infer_s / audio_s if audio_s else float("nan")
    print(f">>> {args.model}/{args.dataset}: 合成 {ok}/{len(rows)} 条 -> {out_dir}  RTF={rtf:.4f}")
    (out_dir / "_rtf.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
