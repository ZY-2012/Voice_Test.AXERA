#!/usr/bin/env python3
"""AISHELL-1(中文) 整理：raw testsets -> asr/aishell1/{wav, wav.scp, text, text.nospace}
raw 由 download_datasets.sh 下到 vad/aishell1/speech_asr_aishell1_testsets（含 wav/test 与 transcript）。
用法: python prepare_aishell.py [--data_root ...]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=cfg["paths"]["data_root"])
    args = ap.parse_args()
    root = Path(args.data_root)
    raw = root / "vad" / "aishell1" / "speech_asr_aishell1_testsets"
    out = root / "asr" / "aishell1"
    if not raw.exists():
        print(f"跳过 AISHELL-1：未找到 {raw}")
        return
    (out / "wav").mkdir(parents=True, exist_ok=True)

    trans = {}
    tf = raw / "transcript" / "aishell_transcript_v0.8.text"
    for line in tf.read_text(encoding="utf-8").splitlines():
        if " " in line:
            uid, t = line.split(" ", 1)
            trans[uid] = t.strip()

    scp, text, nospace = [], [], []
    for w in sorted((raw / "wav" / "test").rglob("*.wav")):
        uid = w.stem
        if uid not in trans:
            continue
        dst = out / "wav" / f"{uid}.wav"
        if not dst.exists():
            shutil.copy(w, dst)
        scp.append(f"{uid}\t{dst.resolve()}")
        text.append(f"{uid}\t{trans[uid]}")
        nospace.append(f"{uid}\t{trans[uid].replace(' ', '')}")
    (out / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")
    (out / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "text.nospace").write_text("\n".join(nospace) + "\n", encoding="utf-8")
    print(f"[ASR] AISHELL-1(zh): {len(scp)} 条 -> {out}")


if __name__ == "__main__":
    main()
