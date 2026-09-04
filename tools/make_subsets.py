#!/usr/bin/env python3
"""确定性子集生成：从已制作的全量数据抽固定子集到 <data_root>/benchmark/。
默认按 configs/benchmark.yaml 的 subset 大小；--full 则全量软链。
子集选择用 (seed, utt_id) 哈希，永远可复现。

输出:
  benchmark/asr/<ds>/{ground_truth.txt, wav/}   # 兼容各模型 test_wer.py
  benchmark/vad/<ds>/subset.scp                 # utt_id\twav\tlabel.npy
  benchmark/se/voicebank_demand/subset.scp      # id\tclean\tnoisy
  benchmark/tts/<ds>/{text, wav.scp}
用法: python tools/make_subsets.py [--module asr vad se tts] [--full]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, deterministic_sample


def _read_scp(path):
    """utt_id<TAB>...  -> dict{uid: rest(list)}"""
    m = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        m[parts[0]] = parts[1:]
    return m


def _symlink(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    # 用相对路径软链：同一挂载在 host(/data/shared/huyuan) 与 板端(/root/huyuan/workspace) 两侧都可解析
    os.symlink(os.path.relpath(os.path.realpath(src), dst.parent), dst)


def sub_asr(data_root, out_root, cfg, full):
    seed = cfg["subset"]["seed"]
    zh_n = cfg["subset"]["asr"]["zh_per_set"]
    en_n = cfg["subset"]["asr"]["en_per_set"]
    zh_sets = {"aishell1"}
    asr_dir = Path(data_root) / "asr"
    for ds_dir in sorted(p for p in asr_dir.iterdir() if p.is_dir()):
        ds = ds_dir.name
        wav_scp, text = ds_dir / "wav.scp", ds_dir / "text"
        if not (wav_scp.exists() and text.exists()):
            continue
        is_zh = ds in zh_sets
        n = None if full else (zh_n if is_zh else en_n)
        # 中文用 nospace 文本作 gt（字符级 CER）
        tf = ds_dir / "text.nospace" if is_zh and (ds_dir / "text.nospace").exists() else text
        wavs = _read_scp(wav_scp)
        texts = _read_scp(tf)
        ids = sorted(set(wavs) & set(texts))
        ids = deterministic_sample(ids, n, seed)
        out = Path(out_root) / "asr" / ds
        (out / "wav").mkdir(parents=True, exist_ok=True)
        gt = []
        for uid in ids:
            wav_path = wavs[uid][0]
            _symlink(wav_path, out / "wav" / f"{uid}.wav")
            gt.append(f"{uid} {texts[uid][0]}")
        (out / "ground_truth.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
        # 各模型 test_wer.py 音频目录约定不一：FireRedASR 用 wav/，Whisper/SenseVoice/WeNet 用 aishell_S0764/
        # 建别名软链使全部通用
        alias = out / "aishell_S0764"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.symlink("wav", alias)
        print(f"  [asr] {ds}: {len(ids)} 条 ({'zh' if is_zh else 'en'})")


def sub_vad(data_root, out_root, cfg, full):
    seed = cfg["subset"]["seed"]
    conf = cfg["subset"]["vad"]
    # ten_official（TEN VAD 官方集）仅 30 条，独立计数（默认即全量）；其余逐句集用 en_utts
    counts = {"librivad/test": conf["en_utts"], "aishell1/test": conf["en_utts"],
              "ten_official/test": conf.get("ten_official_utts", 30)}
    for ds in ["librivad/test", "aishell1/test", "ten_official/test"]:
        scp = Path(data_root) / "vad" / ds / "test.scp"
        if not scp.exists():
            continue
        n = None if full else counts[ds]
        rows = _read_scp(scp)
        ids = deterministic_sample(sorted(rows), n, seed)
        out = Path(out_root) / "vad" / ds.split("/")[0]
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "subset.scp", "w", encoding="utf-8") as f:
            for uid in ids:
                f.write(f"{uid}\t" + "\t".join(rows[uid]) + "\n")
        print(f"  [vad] {ds.split('/')[0]}: {len(ids)} 条")
    # Picovoice 长流：单条，直接引用（不抽样）
    pico = Path(data_root) / "vad" / "picovoice"
    if (pico / "test_audio.wav").exists():
        out = Path(out_root) / "vad" / "picovoice"
        out.mkdir(parents=True, exist_ok=True)
        (out / "info.txt").write_text(
            f"audio={pico/'test_audio.wav'}\nlabels={pico/'benchmark_labels.txt'}\n"
            "frame=512samples(32ms) label:0=sil/1=unknown(ignore)/2=speech\n", encoding="utf-8")
        print("  [vad] picovoice: 长流全量（不抽样）")


def sub_se(data_root, out_root, cfg, full):
    seed = cfg["subset"]["seed"]
    n = None if full else cfg["subset"]["se"]["pairs"]
    scp = Path(data_root) / "se" / "voicebank_demand" / "test.scp"
    if not scp.exists():
        return
    rows = _read_scp(scp)
    ids = deterministic_sample(sorted(rows), n, seed)
    out = Path(out_root) / "se" / "voicebank_demand"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "subset.scp", "w", encoding="utf-8") as f:
        for uid in ids:
            f.write(f"{uid}\t" + "\t".join(rows[uid]) + "\n")
    print(f"  [se] voicebank_demand: {len(ids)} 对")


def sub_tts(data_root, out_root, cfg, full):
    seed = cfg["subset"]["seed"]
    conf = cfg["subset"]["tts"]
    for ds, key in [("ljspeech", "en_utts"), ("aishell3", "zh_utts"),
                    ("librispeech", "en_librispeech"), ("zh_hardcase", "zh_hardcase"),
                    ("zh_long", "zh_long")]:
        d = Path(data_root) / "tts" / ds
        if not (d / "text").exists():
            continue
        n = None if full else conf.get(key)
        texts = _read_scp(d / "text")
        # zh_hardcase 无参考音频（纯文本难例集）：跳过 wav.scp，MCD/GT 锚点不适用
        has_wav = (d / "wav.scp").exists()
        wavs = _read_scp(d / "wav.scp") if has_wav else {}
        ids = sorted(set(wavs) & set(texts)) if has_wav else sorted(texts)
        ids = deterministic_sample(ids, n, seed)
        out = Path(out_root) / "tts" / ds
        out.mkdir(parents=True, exist_ok=True)
        (out / "text").write_text(
            "\n".join(f"{u}\t{texts[u][0]}" for u in ids) + "\n", encoding="utf-8")
        if has_wav:
            (out / "wav.scp").write_text(
                "\n".join(f"{u}\t{wavs[u][0]}" for u in ids) + "\n", encoding="utf-8")
        # text.ref：CER 参考与 TTS 输入不同时（如难例集数字读法）需另存
        if (d / "text.ref").exists():
            refs = _read_scp(d / "text.ref")
            (out / "text.ref").write_text(
                "\n".join(f"{u}\t{refs[u][0]}" for u in ids if u in refs) + "\n",
                encoding="utf-8")
        print(f"  [tts] {ds}: {len(ids)} 条{'' if has_wav else '（无参考音频）'}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", nargs="+", default=["asr", "vad", "se", "tts"])
    ap.add_argument("--full", action="store_true", help="全量（不抽样）")
    args = ap.parse_args()
    dr = cfg["paths"]["data_root"]
    out = str(Path(dr) / "benchmark")
    print(f">>> 生成子集到 {out}  ({'全量' if args.full else '标准子集'})")
    fns = {"asr": sub_asr, "vad": sub_vad, "se": sub_se, "tts": sub_tts}
    for m in args.module:
        fns[m](dr, out, cfg, args.full)
    print(">>> 子集生成完成")


if __name__ == "__main__":
    main()
