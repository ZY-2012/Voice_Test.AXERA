#!/usr/bin/env python3
"""板端批量 ASR 转写：wav 目录 -> hyp.txt（`utt_id<TAB>转写文本`）。

设计要点（为什么不直接用各模型仓库的 test_wer.py）：
  1. vendor test_wer.py 用 `audio_path, gt = line.split(" ")` 解析 ground truth，
     英文转写含空格必然 ValueError 崩溃 —— 英文回环因此从来跑不出结果
  2. vendor 的 "WER" 是对字符串算编辑距离（分母 len(reference)），中文=CER 正确，
     但英文按字符算，标称 WER 实为 CER
  3. 转写与算分解耦后，算分统一走 tools/metrics_asr.py（中文按字 / 英文按词），
     归一化口径单一，且换 ASR 只需换这一步

模型载入一次后批量推理（与 RTF 口径一致：不含加载）。输出 TAB 分隔，天然规避空格歧义。

用法:
  python tts/asr_transcribe.py --asr firered --wav_dir DIR --out hyp.txt --lang zh
  python tts/asr_transcribe.py --asr sensevoice --wav_dir DIR --out hyp.txt --lang en
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, remap  # noqa: E402


def load_firered(mdir, chip):
    """FireRedASR-AED（中文 CER 最优）。长音频由内部 silero VAD 切块，audio_dur 是块大小而非截断上限。"""
    axm = mdir / "axmodel"
    sys.path.insert(0, str(mdir))
    from fireredasr_axmodel import FireRedASRAxModel
    model = FireRedASRAxModel(
        str(axm / "encoder.axmodel"), str(axm / "decoder_loop.axmodel"),
        str(axm / "cmvn.ark"), str(axm / "dict.txt"), str(axm / "train_bpe1000.model"),
        decode_max_len=128, audio_dur=10)

    def transcribe(wav_path, lang):
        # AED 架构无需语言提示（BPE1000 词表覆盖中英），lang 仅记录用
        results, _, _ = model.transcribe([str(wav_path)], 1, 1)
        return results["text"]

    return transcribe, str(mdir)


def load_sensevoice(mdir, chip):
    """SenseVoice（RTF 最低，--language 真实生效）。"""
    mroot = mdir / f"sensevoice_{chip}"
    sys.path.insert(0, str(mdir / "python"))
    from SenseVoiceAx import SenseVoiceAx
    model = SenseVoiceAx(
        str(mroot / "sensevoice.axmodel"), str(mroot / "am.mvn"),
        str(mroot / "tokens.txt"), str(mroot / "chn_jpn_yue_eng_ko_spectok.bpe.model"),
        max_seq_len=256, beam_size=3, hot_words=None, streaming=False)

    def transcribe(wav_path, lang):
        return model.infer(str(wav_path), language=lang, print_rtf=False)

    return transcribe, str(mdir / "python")


# ASR 名 -> (加载器, 模型目录名)
LOADERS = {"firered": (load_firered, "FireRedASR-AED"),
           "sensevoice": (load_sensevoice, "SenseVoice")}


def resolve_model_dir(cfg, name):
    """模型目录：优先本地盘副本（paths.model_local_root/<name>），否则用 NFS 的 model_root。
    NFS 约 2.7MB/s，firered 的 axmodel 有 1.3GB，从 NFS 冷加载需数分钟；本地盘副本可消除该开销。
    支持 ~ 展开（配置里用 ~/asr_local 而非绝对路径，便于他人复现）；不存在时静默回退，
    故 host/板端可共用同一份配置。"""
    import os
    local_root = (cfg["paths"].get("model_local_root") or "").strip()
    if local_root:
        cand = Path(os.path.expanduser(local_root)) / name
        if cand.is_dir():
            return cand, True
    return Path(cfg["paths"]["model_root"]) / name, False


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True, choices=sorted(LOADERS))
    ap.add_argument("--wav_dir", default="", help="待转写 wav 目录（<utt_id>.wav）")
    ap.add_argument("--scp", default="", help="或用 scp: 每行 utt_id<TAB>wav路径（参考音散落多层目录时用）")
    ap.add_argument("--out", required=True, help="输出 hyp 文件（utt_id<TAB>文本）")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--chip", default=cfg["models"]["chip"])
    ap.add_argument("--limit", type=int, default=0, help=">0 时只转写前 N 条")
    ap.add_argument("--ids", default="", help="只转写该文件列出的 utt_id（首列），用于对齐子集")
    args = ap.parse_args()
    if not (args.wav_dir or args.scp):
        ap.error("需指定 --wav_dir 或 --scp")

    # 统一成 [(utt_id, wav_path)]
    if args.scp:
        # scp 里存的是制作时（host 侧）的绝对路径，板端需按挂载对端重映射（common.remap）
        items, missing = [], 0
        for line in Path(args.scp).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, _, rest = line.partition("\t") if "\t" in line else line.partition(" ")
            p = Path(remap(rest.strip()))
            if p.exists():
                items.append((uid, p))
            else:
                missing += 1
        if missing:
            print(f"  [warn] scp 中 {missing} 条音频路径不存在（已跳过）")
    else:
        items = [(p.stem, p) for p in sorted(Path(args.wav_dir).glob("*.wav"))]

    if args.ids and Path(args.ids).exists():
        # 按 ids 文件的**行序**重排（与 synth_batch 的 rows[:limit] 口径一致）
        order = [l.split("\t")[0].split(" ")[0] for l in
                 Path(args.ids).read_text(encoding="utf-8").splitlines() if l.strip()]
        # ⚠️ limit 必须先在**清单**上截断，再与实际存在的音频求交。
        # 若反过来（先取存在的音频再截断前 N），当前 N 条里有合成失败时，
        # 列表会顺移去补靠后的音频，导致转写集与 eval_tts 的参考集不是同一批。
        if args.limit > 0:
            order = order[:args.limit]
        rank = {u: i for i, u in enumerate(order)}
        items = sorted((it for it in items if it[0] in rank), key=lambda it: rank[it[0]])
        miss = len(order) - len(items)
        if miss > 0:
            print(f"  [warn] 清单前 {len(order)} 条中 {miss} 条无对应音频（合成失败/未合成），"
                  f"eval 会按空 hyp 计为删除错误")
    elif args.limit > 0:
        items = items[:args.limit]
    if not items:
        print(f"  [{args.asr}] 无可转写音频（wav_dir={args.wav_dir} scp={args.scp}）")
        sys.exit(2)   # 非 0：让调用方（run.sh）能感知失败，而不是静默跳过

    src = args.scp or args.wav_dir
    print(f">>> {args.asr} 转写 {len(items)} 条（{args.lang}）<- {src}")
    loader, dir_name = LOADERS[args.asr]
    mdir, is_local = resolve_model_dir(cfg, dir_name)
    print(f"    模型目录: {mdir}" + ("（板端本地盘）" if is_local else "（NFS，冷加载较慢）"))
    transcribe, workdir = loader(mdir, args.chip)

    # vendor 模块可能依赖相对路径资源，切到其目录后再推理。
    # 注意：所有路径必须在 chdir **之前**转绝对，否则相对路径会解析到模型目录下
    import os
    items = [(u, p.resolve()) for u, p in items]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    ok = fail = 0
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as f:
        for n, (uid, w) in enumerate(items, 1):
            try:
                hyp = transcribe(w, args.lang)
            except Exception as e:
                hyp = ""
                fail += 1
                print(f"  [{n}/{len(items)}] {uid} 转写失败: {type(e).__name__}: {e}")
            else:
                ok += 1
            # 单行输出：折叠所有空白，避免破坏 TAB 格式
            hyp = " ".join(str(hyp).split())
            f.write(f"{uid}\t{hyp}\n")
            if n % 50 == 0:
                print(f"  [{n}/{len(items)}] ...")
    dt = time.perf_counter() - t0
    print(f">>> 转写完成 {ok} 成功 / {fail} 失败，耗时 {dt:.1f}s -> {out}")


if __name__ == "__main__":
    main()
