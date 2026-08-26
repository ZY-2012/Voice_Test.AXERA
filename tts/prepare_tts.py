#!/usr/bin/env python3
"""TTS 数据制作：
  - LJSpeech(en)：test.txt + metadata.csv -> wav.scp + text
  - AISHELL-3(zh)：test/content.txt -> wav.scp + text（拼音已剥离为纯汉字）
用法: python prepare_tts.py [--data_root ...]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config  # noqa: E402


def prep_ljspeech(base):
    if not (base / "test.txt").exists():
        print(f"跳过 LJSpeech：无 {base/'test.txt'}")
        return
    meta = {}
    for line in (base / "metadata.csv").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            u, norm, _ = (line.split("|") + ["", ""])[:3]
            meta[u] = norm
    ids = [x.strip() for x in (base / "test.txt").read_text().splitlines() if x.strip()]
    text, scp = [], []
    for u in ids:
        w = base / "wav" / f"{u}.wav"
        if u in meta and w.exists():
            text.append(f"{u}\t{meta[u]}")
            scp.append(f"{u}\t{w.resolve()}")
    (base / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (base / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")
    print(f"[TTS] LJSpeech(en): {len(scp)} 条 -> {base}")


def prep_aishell3(base):
    content = base / "test" / "content.txt"
    if not content.exists():
        print(f"跳过 AISHELL-3：无 {content}")
        return
    tmap = {}
    for line in content.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            wav, rest = line.split("\t", 1)
            uid = wav.replace(".wav", "")
            tmap[uid] = re.sub(r"\s+", "", re.sub(r"[a-z]+[1-5]?", "", rest))  # 去拼音
    wavs = sorted((base / "test" / "wav").rglob("*.wav"))
    text, scp = [], []
    for w in wavs:
        if w.stem in tmap:
            text.append(f"{w.stem}\t{tmap[w.stem]}")
            scp.append(f"{w.stem}\t{w.resolve()}")
    (base / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (base / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")
    spk = len(set(w.stem[:7] for w in wavs))
    print(f"[TTS] AISHELL-3(zh): {len(scp)} 条 / {spk} 说话人 (44.1kHz) -> {base}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=cfg["paths"]["data_root"])
    args = ap.parse_args()
    root = Path(args.data_root) / "tts"
    prep_ljspeech(root / "ljspeech")
    prep_aishell3(root / "aishell3")


if __name__ == "__main__":
    main()
