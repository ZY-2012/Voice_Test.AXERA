#!/usr/bin/env python3
"""TTS 评测（板端）：把转写结果与输入文本比对，产出可解读的客观指标。

指标：
  - 回环 CER/WER：合成音经板端 ASR 转写（--asr_hyp: id<TAB>hyp）vs 输入文本（可懂度）
  - S/D/I 分解：del_rate 高=截断/漏读，ins_rate 高=重复/幻听，sub_rate 高=发音错
  - len_ratio：合成音时长 / 参考音时长（不依赖 ASR 的独立故障探针）
  - success_rate：合成成功条数 / 应合成条数
  - MCD：合成音 vs 参考音，DTW 对齐梅尔倒谱距离（仅同参考说话人可比）
  - GT 锚点（--gt_anchor）：把参考真人音频的转写当作 hyp，得 ASR 地板；
    模型名写 GT，用于扣除 ASR 自身误差后解读 TTS 数字

输出行追加到 results/tts.csv。
用法:
  python eval_tts.py --dataset aishell3 --syn_dir DIR [--asr_hyp hyp.txt] [--model_name melotts]
  python eval_tts.py --dataset aishell3 --gt_anchor --asr_hyp gt_hyp.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, remap, TTS_DATASETS, tts_lang
import metrics_tts as T
from metrics_asr import score as asr_score, load_text


def wav_duration(path):
    import soundfile as sf
    try:
        return sf.info(str(path)).duration
    except Exception:
        return None


def _ref_path(v):
    """wav.scp 的值 -> 本侧可用路径（host/board 挂载重映射）。"""
    return remap(v[0] if isinstance(v, list) else v)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TTS_DATASETS))
    ap.add_argument("--syn_dir", default="", help="TTS 合成音频目录 <id>.wav（GT 锚点模式可不给）")
    ap.add_argument("--asr_hyp", default="", help="转写结果 id<TAB>hyp")
    ap.add_argument("--model_name", default="tts-model")
    ap.add_argument("--gt_anchor", action="store_true",
                    help="GT 锚点模式：asr_hyp 是参考真人音频的转写，指标记为 *_gt，模型名 GT")
    ap.add_argument("--asr_name", default="", help="转写所用 ASR，写入 note 列")
    ap.add_argument("--mcd", action="store_true",
                    help="额外算 MCD（默认关闭：仅同说话人可比，预设音色模型跨说话人无意义）")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只按 text 前 N 条算指标（与 synth_batch --limit 同口径，冒烟用）")
    args = ap.parse_args()
    lang = tts_lang(args.dataset)
    bench = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset
    # CER 参考优先用 text.ref：难例集数字类的 TTS 输入是书面形式（2026年），
    # 而 ASR 转写是口语形式（二零二六年），必须用口语形式作参考
    text_ref = bench / "text.ref" if (bench / "text.ref").exists() else bench / "text"
    text_in = bench / "text"
    wav_scp = bench / "wav.scp"
    if not text_in.exists():
        print(f"无子集 {text_in}"); return
    if text_ref.name == "text.ref":
        print(f"  参考文本用 {text_ref.name}（TTS 输入与 CER 参考不同）")

    csv = REPO / "results" / "tts.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        csv.write_text("module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note\n")
    out = []
    asr_note = f"ASR={args.asr_name or cfg['eval']['tts'].get('asr_model', '')}"
    model_name = "GT" if args.gt_anchor else args.model_name

    # ---- 回环可懂度 + S/D/I 分解（GT 锚点模式下指标名加 _gt 后缀）----
    # 注意 CSV 约定：速率类指标存**百分数**（如 cer_loopback=70.59），report.py 直接加 % 渲染；
    # len_ratio/mcd 存原值。metrics_asr 返回 0~1 分数，故此处 ×100。
    ref_texts = load_text(text_ref)
    in_texts = load_text(text_in)
    if args.limit > 0:
        # 与 synth_batch 的 rows[:limit] 同口径：按 text 行序取前 N
        keep = list(in_texts)[:args.limit]
        in_texts = {u: in_texts[u] for u in keep}
        ref_texts = {u: ref_texts[u] for u in keep if u in ref_texts}
        print(f"  --limit {args.limit}：只统计前 {len(in_texts)} 条")
    if args.asr_hyp and Path(args.asr_hyp).exists():
        r = asr_score(load_text(args.asr_hyp), ref_texts, lang=lang)
        m = r["metric"]                                  # zh->cer / en->wer
        suffix = "_gt" if args.gt_anchor else "_loopback"
        out.append((f"{m}{suffix}", r[m] * 100, f"{asr_note};{r['num_utt']}条"))
        for k in ("sub_rate", "del_rate", "ins_rate"):
            out.append((k + ("_gt" if args.gt_anchor else ""), r[k] * 100, asr_note))
        tag = "GT锚点" if args.gt_anchor else "回环"
        print(f"  {tag} {m.upper()}={r[m]*100:.2f}%  "
              f"(替换{r['sub_rate']*100:.2f}% 删除{r['del_rate']*100:.2f}% 插入{r['ins_rate']*100:.2f}%)")
        if r["num_missing_hyp"]:
            print(f"    注意：{r['num_missing_hyp']} 条无转写（按空 hyp 计入，全额删除错误）")

    ref_map = load_text(wav_scp) if wav_scp.exists() else {}

    # ---- 合成侧指标（GT 锚点模式跳过：没有合成音）----
    if not args.gt_anchor and args.syn_dir:
        syn_dir = Path(args.syn_dir)
        n_expect = len(in_texts)                # 应合成条数按 TTS 输入清单（已含 limit）算
        syn_wavs = {p.stem: p for p in syn_dir.glob("*.wav") if p.stat().st_size > 0}
        syn_wavs = {u: p for u, p in syn_wavs.items() if u in in_texts}
        if n_expect:
            out.append(("success_rate", len(syn_wavs) / n_expect * 100,
                        f"{len(syn_wavs)}/{n_expect}条"))
            print(f"  合成成功率={len(syn_wavs)/n_expect*100:.1f}% ({len(syn_wavs)}/{n_expect})")

        # 合成音平均时长：固定 shape 的 axmodel 每条推理耗时近似恒定，
        # RTF 被音频长短主导（短句 RTF 虚高）。必须与 RTF 同表展示才可解读。
        durs = [d for d in (wav_duration(p) for p in syn_wavs.values()) if d]
        if durs:
            out.append(("audio_avg_s", float(np.mean(durs)), f"{len(durs)}条合成音均长"))
            print(f"  合成音平均时长={np.mean(durs):.2f}s（RTF 解读须结合此值）")

        # len_ratio：合成音时长 / 参考音时长（无参考音则跳过）
        ratios = []
        for uid, ref in ref_map.items():
            if uid not in syn_wavs:
                continue
            d_ref, d_syn = wav_duration(_ref_path(ref)), wav_duration(syn_wavs[uid])
            if d_ref and d_syn:
                ratios.append(d_syn / d_ref)
        if ratios:
            out.append(("len_ratio", float(np.mean(ratios)), f"{len(ratios)}对"))
            print(f"  时长比={np.mean(ratios):.3f} ({len(ratios)}对，显著<1=截断 >>1=拖尾)")
        elif ref_map:
            print("  [warn] 时长比无结果：参考音频路径解析失败（检查 wav.scp 与挂载映射）")

        # MCD（需参考音；默认关闭——仅同说话人有意义，预设音色模型跨说话人无解释力）
        if args.mcd:
            vals = []
            for uid, ref in ref_map.items():
                if uid not in syn_wavs:
                    continue
                try:
                    vals.append(T.mcd(_ref_path(ref), str(syn_wavs[uid])))
                except Exception:
                    pass
            if vals:
                out.append(("mcd", float(np.mean(vals)), f"{len(vals)}对;仅同说话人可比"))
                print(f"  MCD={np.mean(vals):.3f} dB ({len(vals)}对)")

    if not out:
        print("  无可写指标（检查 --asr_hyp / --syn_dir）"); return
    with open(csv, "a") as f:
        for metric, val, note in out:
            f.write(f"tts,{model_name},{args.dataset},{lang},{metric},{val:.4f},,,,{note}\n")
    print(f">>> TTS 评测完成 {len(out)} 条指标 -> {csv}")


if __name__ == "__main__":
    main()
