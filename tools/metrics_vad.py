#!/usr/bin/env python3
"""VAD 指标：帧级 accuracy/precision/recall/F1 + ROC-AUC + 事件级端点误差。

支持两种参考标签：
  - 采样级 0/1（LibriVAD / AISHELL 的 .npy）
  - 帧级 0/1/2（Picovoice：0=静音 1=未知[忽略] 2=语音）

预测输入统一为「每帧语音概率或二值」。提供 segments→帧 / 采样→帧 转换helper。
"""
import numpy as np


def samples_to_frames(sample_labels, sr=16000, frame_ms=32, thresh=0.5):
    """采样级 0/1 → 帧级 0/1：每帧内语音采样占比 >= thresh 记为语音。"""
    hop = int(sr * frame_ms / 1000)
    n = len(sample_labels) // hop
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    frames = sample_labels[:n * hop].reshape(n, hop)
    return (frames.mean(axis=1) >= thresh).astype(np.int8)


def segments_to_frames(segments_ms, total_frames, frame_ms=32):
    """语音段 [(start_ms,end_ms),...] → 帧级 0/1。"""
    lab = np.zeros(total_frames, dtype=np.int8)
    for s, e in segments_ms:
        i, j = int(s / frame_ms), int(np.ceil(e / frame_ms))
        lab[max(0, i):min(total_frames, j)] = 1
    return lab


def prob_to_frames(probs, total_frames):
    """把任意长度的每帧概率线性重采样到 total_frames（对齐参考帧数）。"""
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == total_frames:
        return probs
    if len(probs) == 0:
        return np.zeros(total_frames)
    idx = np.linspace(0, len(probs) - 1, total_frames)
    return np.interp(idx, np.arange(len(probs)), probs)


def frame_scores(pred_prob, ref_label, ignore_label=None, threshold=0.5):
    """单条：帧级混淆矩阵指标。
    pred_prob: 每帧语音概率(float) 或二值; ref_label: 每帧 0/1(/2); 长度需一致。
    ignore_label: 需忽略的参考标签值（如 Picovoice 的 1=未知）。
    返回 (tp, fp, tn, fn) 计数。"""
    pred = (np.asarray(pred_prob) >= threshold).astype(np.int8)
    ref = np.asarray(ref_label)
    # Picovoice: 2=语音→1, 0=静音→0, 1=未知→忽略
    mask = np.ones(len(ref), dtype=bool)
    if ignore_label is not None:
        mask = ref != ignore_label
    ref_bin = (ref == 2).astype(np.int8) if (ref == 2).any() else ref
    p, r = pred[mask], ref_bin[mask]
    tp = int(((p == 1) & (r == 1)).sum())
    fp = int(((p == 1) & (r == 0)).sum())
    tn = int(((p == 0) & (r == 0)).sum())
    fn = int(((p == 0) & (r == 1)).sum())
    return tp, fp, tn, fn


def prf(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def roc_auc(pairs):
    """pairs: [(pred_prob_array, ref_bin_array), ...]（已去忽略帧）。
    扫阈值算 (FPR,TPR) → 梯形积分 AUC。"""
    probs = np.concatenate([p for p, _ in pairs])
    refs = np.concatenate([r for _, r in pairs])
    P = (refs == 1).sum()
    N = (refs == 0).sum()
    if P == 0 or N == 0:
        return float("nan"), []
    order = np.argsort(-probs)
    refs = refs[order]
    tp = np.cumsum(refs == 1)
    fp = np.cumsum(refs == 0)
    tpr = np.concatenate([[0], tp / P])
    fpr = np.concatenate([[0], fp / N])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # numpy2.0 改名
    auc = float(_trapz(tpr, fpr))
    return auc, list(zip(fpr.tolist(), tpr.tolist()))


def aggregate(items, ignore_label=None, threshold=0.5, frame_ms=32):
    """items: [(pred_prob_frames, ref_label_frames), ...]（长度对齐）。
    返回帧级 P/R/F1/acc（阈值）+ ROC-AUC（全阈值）。"""
    TP = FP = TN = FN = 0
    roc_pairs = []
    for pred, ref in items:
        ref = np.asarray(ref)
        mask = np.ones(len(ref), dtype=bool)
        if ignore_label is not None:
            mask = ref != ignore_label
        ref_bin = (ref == 2).astype(np.int8) if (ref == 2).any() else ref
        tp, fp, tn, fn = frame_scores(pred, ref, ignore_label, threshold)
        TP += tp; FP += fp; TN += tn; FN += fn
        roc_pairs.append((np.asarray(pred)[mask], ref_bin[mask]))
    res = prf(TP, FP, TN, FN)
    res["roc_auc"], _ = roc_auc(roc_pairs)
    res["counts"] = {"tp": TP, "fp": FP, "tn": TN, "fn": FN}
    res["threshold"] = threshold
    return res
