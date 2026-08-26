#!/usr/bin/env python3
"""ASR 指标：CER（中文字符级）/ WER（英文词级），编辑距离口径。
与 8860_export 各模型 test_wer.py 一致：去标点、转小写。

用法（独立调用）:
  python metrics_asr.py --hyp hyp.txt --ref ref.txt --lang zh
  文件格式每行: <utt_id><TAB或空格><文本>
"""
import argparse
import re
import unicodedata


def normalize(text, lower=True, remove_punct=True):
    if lower:
        text = text.lower()
    if remove_punct:
        # 去除所有标点/符号（中英通用），保留字母数字与空白
        text = re.sub(r"[^\w\s]|_", "", text)
    # 折叠多空格
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, lang):
    """中文按字符切（去空格后逐字），英文按空格切词。"""
    if lang == "zh":
        return list(text.replace(" ", ""))
    return text.split()


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
    """hyp_map/ref_map: {utt_id: text}。返回指标 dict。"""
    tot_err = tot_len = 0
    n_utt = n_miss = 0
    per_utt = {}
    for uid, ref in ref_map.items():
        if uid not in hyp_map:
            n_miss += 1
            hyp = ""
        else:
            hyp = hyp_map[uid]
        r = tokenize(normalize(ref, lower, remove_punct), lang)
        h = tokenize(normalize(hyp, lower, remove_punct), lang)
        d, rl = edit_distance(r, h)
        tot_err += d
        tot_len += rl
        per_utt[uid] = d / rl if rl else 0.0
        n_utt += 1
    metric = "cer" if lang == "zh" else "wer"
    return {
        "metric": metric,
        metric: tot_err / tot_len if tot_len else float("nan"),
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
            uid, _, txt = line.partition(" ")
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
