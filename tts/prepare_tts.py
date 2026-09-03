#!/usr/bin/env python3
"""TTS 数据制作：
  - LJSpeech(en)：test.txt + metadata.csv -> wav.scp + text
  - AISHELL-3(zh)：test/content.txt -> wav.scp + text（拼音已剥离为纯汉字）
  - LibriSpeech(en)：复用 asr/librispeech，按说话人过滤出 test-clean（多说话人）
  - zh_hardcase(zh)：仓库内 tts/zh_hardcase.tsv -> text + text.ref（无参考音频）
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


def prep_librispeech(base, data_root):
    """LibriSpeech test-clean（英文多说话人）：复用 asr/librispeech/，按说话人 ID 过滤出 test-clean。
    asr 目录混放 test-clean(2620)+test-other(2939)，用仓库内 librispeech_testclean_spk.txt 过滤。"""
    src = Path(data_root) / "asr" / "librispeech"
    spk_f = Path(__file__).resolve().parent / "librispeech_testclean_spk.txt"
    if not (src / "wav.scp").exists() or not spk_f.exists():
        print(f"跳过 LibriSpeech：需 {src/'wav.scp'} 与 {spk_f.name}")
        return
    spks = {l.strip() for l in spk_f.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}
    texts, wavs = {}, {}
    for line in (src / "text").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            u, t = line.split("\t", 1)
            texts[u] = t
    for line in (src / "wav.scp").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            u, w = line.split("\t", 1)
            wavs[u] = w
    ids = sorted(u for u in set(texts) & set(wavs) if u.split("-")[0] in spks)
    base.mkdir(parents=True, exist_ok=True)
    (base / "text").write_text("\n".join(f"{u}\t{texts[u]}" for u in ids) + "\n", encoding="utf-8")
    (base / "wav.scp").write_text("\n".join(f"{u}\t{wavs[u]}" for u in ids) + "\n", encoding="utf-8")
    print(f"[TTS] LibriSpeech test-clean(en): {len(ids)} 条 / {len(spks)} 说话人 "
          f"(16kHz，复用 asr/librispeech) -> {base}")


def _read_hardcase_tsv():
    """读仓库内难例定义 -> [(uid, 输入文本, CER参考文本)]。"""
    src = Path(__file__).resolve().parent / "zh_hardcase.tsv"
    if not src.exists():
        return []
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 2 or not p[0] or not p[1]:
            continue
        uid, inp = p[0], p[1]
        rows.append((uid, inp, p[2] if len(p) > 2 and p[2].strip() else inp))
    return rows


def _write_text_pair(base, rows, label):
    base.mkdir(parents=True, exist_ok=True)
    (base / "text").write_text("\n".join(f"{u}\t{i}" for u, i, _ in rows) + "\n", encoding="utf-8")
    (base / "text.ref").write_text("\n".join(f"{u}\t{r}" for u, _, r in rows) + "\n", encoding="utf-8")
    print(f"[TTS] {label}: {len(rows)} 条（无参考音频，不适用 MCD/GT 锚点）-> {base}")


def prep_zh_hardcase(base):
    """难例集：仓库内 tts/zh_hardcase.tsv（版本化的测试定义）-> data_root/tts/zh_hardcase/。
    产出 text（TTS 输入）+ text.ref（CER 参考，数字类为口语形式）。无参考音频，故无 wav.scp。"""
    rows = _read_hardcase_tsv()
    if not rows:
        print("跳过 zh_hardcase：无 tts/zh_hardcase.tsv")
        return
    _write_text_pair(base, rows, "zh_hardcase(zh)")


def prep_zh_long(base):
    """长句集：难例集里的 hard_long_* 单独成集，作 **RTF 主口径的测量集**。
    原因：固定 shape 的 axmodel 每条推理耗时近似恒定，RTF 被音频长短主导 ——
    AISHELL-3 均长仅约 1.3s 会让 RTF 虚高数倍，与各仓库按长句/段落报的官方值不可比。"""
    rows = [r for r in _read_hardcase_tsv() if r[0].startswith("hard_long")]
    if not rows:
        print("跳过 zh_long：难例集里无 hard_long_* 条目")
        return
    _write_text_pair(base, rows, "zh_long(zh, RTF 基准集)")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=cfg["paths"]["data_root"])
    args = ap.parse_args()
    root = Path(args.data_root) / "tts"
    prep_ljspeech(root / "ljspeech")
    prep_aishell3(root / "aishell3")
    prep_librispeech(root / "librispeech", args.data_root)
    prep_zh_hardcase(root / "zh_hardcase")
    prep_zh_long(root / "zh_long")


if __name__ == "__main__":
    main()
