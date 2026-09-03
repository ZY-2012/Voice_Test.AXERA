#!/usr/bin/env python3
"""ASR 指标：CER（中文字符级）/ WER（英文词级），编辑距离口径。

归一化口径（本文件是唯一事实来源，不依赖各模型 vendor test_wer.py 的内部实现）：
  1. 剥离 ASR 富文本标签（SenseVoice 的 <|zh|> <|HAPPY|> <|withitn|> 等）—— 必须在去标点前做，
     否则去掉 <|> 后 zh/HAPPY 会残留成假词污染指标
  2. 中文：繁体 → 简体（zhconv）
  3. 转小写、去标点、折叠空白
  4. 切分：中文按字、英文按空格切词

除总错误率外还给出 S/D/I 分解：删除率高=截断/漏读，插入率高=重复/幻听，替换率高=发音错。

用法（独立调用）:
  python metrics_asr.py --hyp hyp.txt --ref ref.txt --lang zh
  文件格式每行: <utt_id><TAB或空格><文本>
"""
import argparse
import re
import sys

# ASR 富文本标签，如 <|zh|> <|NEUTRAL|> <|Speech|> <|withitn|>
_TAG_RE = re.compile(r"<\|[^|>]*\|>")
_PUNCT_RE = re.compile(r"[^\w\s]|_")

_zhconv_warned = False


def strip_tags(text):
    """剥离 ASR 富文本标签（语言/情感/事件/ITN 标记）。"""
    return _TAG_RE.sub(" ", text)


def to_simplified(text):
    """繁体 → 简体。zhconv 缺失时告警一次并原样返回（指标会偏高）。"""
    global _zhconv_warned
    try:
        from zhconv import convert
    except ImportError:
        if not _zhconv_warned:
            print("  [warn] 未安装 zhconv，跳过繁→简转换，中文 CER 可能偏高（pip install zhconv）",
                  file=sys.stderr)
            _zhconv_warned = True
        return text
    return convert(text, "zh-cn")


def normalize(text, lang="zh", lower=True, remove_punct=True):
    text = strip_tags(text)          # 必须先剥标签，再去标点
    if lang == "zh":
        text = to_simplified(text)
    if lower:
        text = text.lower()
    if remove_punct:
        # 去除所有标点/符号（中英通用），保留字母数字与空白
        text = _PUNCT_RE.sub("", text)
    # 折叠多空格
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, lang):
    """中文按字符切（去空格后逐字），英文按空格切词。"""
    if lang == "zh":
        return list(text.replace(" ", ""))
    return text.split()


def align_ops(ref, hyp):
    """Levenshtein 对齐并统计操作数。返回 (替换, 删除, 插入)。
    删除 = 参考中有、识别结果里没有（漏读/截断）；插入 = 识别结果多出（重复/幻听）。"""
    n, m = len(ref), len(hyp)
    # 全量 DP 表（需回溯，不能只留两行）
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # 回溯统计操作类型
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1                      # 匹配
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def edit_distance(ref, hyp):
    """Levenshtein 距离（token 序列）。返回 (距离, 参考长度)。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n


def score(hyp_map, ref_map, lang="zh", lower=True, remove_punct=True):
    """hyp_map/ref_map: {utt_id: text}。返回指标 dict（含 S/D/I 分解）。"""
    tot_err = tot_len = 0
    tot_sub = tot_del = tot_ins = 0
    n_utt = n_miss = 0
    per_utt = {}
    for uid, ref in ref_map.items():
        if uid not in hyp_map:
            n_miss += 1
            hyp = ""
        else:
            hyp = hyp_map[uid]
        r = tokenize(normalize(ref, lang=lang, lower=lower, remove_punct=remove_punct), lang)
        h = tokenize(normalize(hyp, lang=lang, lower=lower, remove_punct=remove_punct), lang)
        s, dl, i_ = align_ops(r, h)
        d, rl = s + dl + i_, len(r)
        tot_err += d
        tot_len += rl
        tot_sub += s
        tot_del += dl
        tot_ins += i_
        per_utt[uid] = d / rl if rl else 0.0
        n_utt += 1
    metric = "cer" if lang == "zh" else "wer"
    rate = (lambda x: x / tot_len if tot_len else float("nan"))
    return {
        "metric": metric,
        metric: rate(tot_err),
        "sub_rate": rate(tot_sub),
        "del_rate": rate(tot_del),
        "ins_rate": rate(tot_ins),
        "total_err": tot_err,
        "total_ref_len": tot_len,
        "num_utt": n_utt,
        "num_missing_hyp": n_miss,
        "per_utt": per_utt,
    }


def load_text(path):
    m = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        if "\t" in line:
            uid, txt = line.split("\t", 1)
        else:
            uid, _, txt = line.partition(" ")   # maxsplit=1：英文文本含空格也不会截断
        m[uid] = txt
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    args = ap.parse_args()
    res = score(load_text(args.hyp), load_text(args.ref), args.lang)
    m = res["metric"].upper()
    print(f"{m}: {res[res['metric']]*100:.2f}%  "
          f"({res['total_err']}/{res['total_ref_len']}, {res['num_utt']} utts, "
          f"{res['num_missing_hyp']} missing hyp)")
    print(f"  替换 {res['sub_rate']*100:.2f}%  删除 {res['del_rate']*100:.2f}%  "
          f"插入 {res['ins_rate']*100:.2f}%")
