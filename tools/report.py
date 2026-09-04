#!/usr/bin/env python3
"""结果汇总渲染器（唯一指标来源 = results/*.csv）。
读全部 <module>.csv → 透视成"一行一模型"的美观表 → 生成 results/summary.md，
并可注入顶层 README 的标记区块。所有文档指标数字均由本脚本生成，勿手写。

用法:
  python tools/report.py                # 只写 results/summary.md
  python tools/report.py --readme       # 额外注入 README 的 <!-- RESULTS:* --> / <!-- STATUS-MATRIX -->
  python tools/report.py --readme --dry-run   # 预览注入，不改文件
  python tools/report.py --only asr     # 只渲染某模块（调试）
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, module_names, module_label, REPO_ROOT

CSV_HEADER = ["module", "model", "dataset", "lang", "metric", "value",
              "rtf", "cmm_mb", "os_mb", "note"]


@dataclass
class Fmt:
    label: str          # 列显示名
    digits: int         # 小数位（-1=取整）
    unit: str           # 后缀单位
    lower_better: bool   # 越低越好；None=非单调指标（如时长比越接近 1 越好），不标记最优


# 指标 → 显示规则；未列出的兜底 3 位小数、越高越好
METRIC_FORMATS = {
    "cer":           Fmt("CER", 2, "%", True),
    "wer":           Fmt("WER", 2, "%", True),
    "cer_loopback":  Fmt("回环CER", 2, "%", True),
    "wer_loopback":  Fmt("回环WER", 2, "%", True),
    "cer_gt":        Fmt("GT锚点CER", 2, "%", True),
    "wer_gt":        Fmt("GT锚点WER", 2, "%", True),
    "sub_rate":      Fmt("替换率", 2, "%", True),
    "del_rate":      Fmt("删除率", 2, "%", True),
    "ins_rate":      Fmt("插入率", 2, "%", True),
    "sub_rate_gt":   Fmt("替换率(GT)", 2, "%", True),
    "del_rate_gt":   Fmt("删除率(GT)", 2, "%", True),
    "ins_rate_gt":   Fmt("插入率(GT)", 2, "%", True),
    "len_ratio":     Fmt("时长比", 3, "", None),       # 越接近 1 越好（非单调，不标最优）
    "audio_avg_s":   Fmt("均长", 2, "s", None),        # 合成音平均时长；RTF 解读的必要上下文
    "success_rate":  Fmt("成功率", 1, "%", False),
    "f1":            Fmt("F1", 3, "", False),
    "accuracy":      Fmt("准确率", 3, "", False),
    "precision":     Fmt("精确率", 3, "", False),
    "recall":        Fmt("召回率", 3, "", False),
    "roc_auc":       Fmt("AUC", 3, "", False),
    "stoi":          Fmt("STOI", 3, "", False),
    "pesq":          Fmt("PESQ", 2, "", False),
    "si_snr":        Fmt("SI-SNR", 1, " dB", False),
    "mcd":           Fmt("MCD", 2, " dB", True),
    "rtf":           Fmt("RTF", 3, "", True),      # 独立列（不在 metric 中）
    "rtf_e2e":       Fmt("RTF(含加载)", 2, "", True),  # synth_batch 逐条子进程口径，与 RTF 主口径不可混比
    "cmm_mb":        Fmt("CMM(MB)", 1, "", True),   # 常在 1MB 以下，取整会全变 0/1
    "os_mb":         Fmt("OS(MB)", -1, "", True),
}
_DEFAULT_FMT = Fmt("", 3, "", False)


def fmt_value(v, spec):
    """格式化数值；空/非数值 → —。"""
    if v is None or v == "":
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    s = f"{round(x)}" if spec.digits < 0 else f"{x:.{spec.digits}f}"
    return s + spec.unit


def load_rows(results_dir):
    """读全部 <module>.csv（校验表头），返回记录列表。"""
    rows = []
    for csvf in sorted(Path(results_dir).glob("*.csv")):
        with open(csvf, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != CSV_HEADER:
                print(f"  [warn] {csvf.name} 表头非标准（{rd.fieldnames}），跳过")
                continue
            rows += [r for r in rd if r.get("model")]
    return rows


def pivot(module_rows):
    """把某模块的记录透视成 (model,dataset) 行：
    返回 (records, metric_cols, has_multi_dataset, extra_cols)。
    每条 record: {model, dataset, <metric>: value..., rtf, cmm_mb, os_mb, note}

    同一 (model,dataset,metric) 出现多行时**取最后一行**（CSV 追加写，故最后 = 最新口径）。
    典型场景：同模型先后用不同评测 ASR 跑过（如 sensevoice 旧口径 → firered 新口径），
    两套数值都保留在 CSV 里，表格只展示最新的；被覆盖的会打印告警以免误读。"""
    order = []          # (model,dataset) 保持首见顺序
    rec = {}
    metric_cols = []
    shadowed = []       # 被后来行覆盖的旧值（口径变更时提示）
    # CSV 既有 rtf/cmm_mb/os_mb 独立字段，又允许用 metric 列写同名指标（新旧两种写法并存）。
    # 同名时归并到同一列，否则会渲染出两个 RTF 列。
    FIELD_METRICS = ("rtf", "cmm_mb", "os_mb")
    for r in module_rows:
        key = (r["model"], r["dataset"])
        if key not in rec:
            rec[key] = {"model": r["model"], "dataset": r["dataset"],
                        "lang": r.get("lang", ""),
                        "rtf": r.get("rtf", ""), "cmm_mb": r.get("cmm_mb", ""),
                        "os_mb": r.get("os_mb", ""), "note": r.get("note", "")}
            order.append(key)
        m = r["metric"]
        if m in FIELD_METRICS:
            rec[key][m] = r.get("value", "")     # 归并到同名字段列，不新增 metric 列
            continue
        if m not in metric_cols:
            metric_cols.append(m)
        if m in rec[key] and rec[key][m] != r.get("value", ""):
            shadowed.append((r["model"], r["dataset"], m, rec[key][m], r.get("value", "")))
        rec[key][m] = r.get("value", "")
        # rtf/内存取该模型首个非空
        for c in FIELD_METRICS:
            if not rec[key].get(c) and r.get(c):
                rec[key][c] = r[c]
    for model, ds, m, old, new in shadowed:
        print(f"  [info] {model}/{ds} 的 {m} 有多个口径：表格用最新值 {new}（旧值 {old} 仍保留在 CSV）")
    records = [rec[k] for k in order]
    has_multi_ds = len({d for _, d in order}) > 1
    extra = [c for c in ("rtf", "cmm_mb", "os_mb")
             if any(rec[k].get(c) for k in order)]
    return records, metric_cols, has_multi_ds, extra


def _best_index(records, col, spec):
    """该列最优行索引（≥2 行才返回，否则 None）。非单调指标（lower_better=None）不标记。"""
    if spec.lower_better is None:
        return None
    vals = []
    for i, rc in enumerate(records):
        try:
            vals.append((i, float(rc.get(col, ""))))
        except (TypeError, ValueError):
            pass
    if len(vals) < 2:
        return None
    return (min if spec.lower_better else max)(vals, key=lambda t: t[1])[0]


def _sort_by_primary(records, prim, specs, metric_cols):
    """按主指标排序（缺值排末尾）。"""
    if prim not in metric_cols:
        return records
    ps = specs[prim]

    def _k(rc):
        try:
            return (0, float(rc.get(prim, "")) * (1 if ps.lower_better else -1))
        except (TypeError, ValueError):
            return (1, 0.0)

    return sorted(records, key=_k)


def _md_table(head, rows):
    return ["| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] * len(head)) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def _render_rows(records, cols, specs):
    """一行一模型的表；每列最优加粗——只在本表（同一数据集）内比较。"""
    best = {c: _best_index(records, c, specs[c]) for c in cols}
    rows = []
    for i, rc in enumerate(records):
        cells = [rc["model"]]
        for c in cols:
            txt = fmt_value(rc.get(c, ""), specs[c])
            if best[c] == i and txt != "—":
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append(cells)
    return _md_table(["模型"] + [specs[c].label or c for c in cols], rows), best


def _primary_matrix(records, datasets, prim, spec):
    """主指标矩阵：行=模型，列=数据集。同一列即同一数据集，纵向直接比模型。
    整行/整列没有主指标值的都不进表（如英文集只有 WER 没有 CER）。"""
    cell = {(rc["model"], rc["dataset"]): rc.get(prim, "") for rc in records}

    def _num(m, ds):
        try:
            return float(cell.get((m, ds), ""))
        except (TypeError, ValueError):
            return None

    all_models = []
    for rc in records:
        if rc["model"] not in all_models:
            all_models.append(rc["model"])
    datasets = [ds for ds in datasets if any(_num(m, ds) is not None for m in all_models)]
    models = [m for m in all_models if any(_num(m, ds) is not None for ds in datasets)]
    if len(datasets) < 2 or not models:
        return []
    if spec.lower_better is not None:       # 按主指标均值排（跨数据集的总体强弱）
        def _mean(m):
            v = [x for ds in datasets if (x := _num(m, ds)) is not None]
            return sum(v) / len(v)
        models.sort(key=_mean, reverse=not spec.lower_better)
    best_of = {}                            # 每个数据集（列）内的最优模型
    for ds in datasets:
        vals = [(m, _num(m, ds)) for m in models if _num(m, ds) is not None]
        if len(vals) >= 2 and spec.lower_better is not None:
            best_of[ds] = (min if spec.lower_better else max)(vals, key=lambda t: t[1])[0]
    rows = []
    for m in models:
        cells = [m]
        for ds in datasets:
            txt = fmt_value(cell.get((m, ds), ""), spec)
            if best_of.get(ds) == m and txt != "—":
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append(cells)
    arrow = "↓" if spec.lower_better else "↑"
    return _md_table([f"{spec.label or prim} {arrow}"] + list(datasets), rows)


def render_module_table(module, module_rows, cfg):
    records, metric_cols, has_multi_ds, extra = pivot(module_rows)
    if not records:
        return ""
    cols = metric_cols + extra                      # 指标列 + rtf/内存
    specs = {c: METRIC_FORMATS.get(c, _DEFAULT_FMT) for c in cols}
    prim = (cfg.get("report", {}).get("primary_metric", {}) or {}).get(module)

    scope = _fill_scope(module, cfg)
    out = [f"**{module_label(module, cfg)}**"]
    if scope:
        out.append(f"> 口径：{scope}")
    out.append("")

    if not has_multi_ds:                            # 单数据集：一张表就够
        records = _sort_by_primary(records, prim, specs, metric_cols)
        lines, best = _render_rows(records, cols, specs)
        out += lines
        if prim in metric_cols and best.get(prim) is not None:
            bi = best[prim]
            out += ["", f"> 最优：{records[bi]['model']} "
                        f"{specs[prim].label}={fmt_value(records[bi].get(prim, ''), specs[prim])}"]
        return "\n".join(out)

    # 多数据集：先主指标矩阵（同一列=同一数据集，纵向直接比模型），再按数据集分表列全指标。
    # 跨数据集混在一张表里比不了——加粗的"最优"会落到别的数据集的行上。
    datasets = []
    for rc in records:                              # 保持首见顺序
        if rc["dataset"] not in datasets:
            datasets.append(rc["dataset"])
    if prim in metric_cols and len(datasets) > 1:
        matrix = _primary_matrix(records, datasets, prim, specs[prim])
        if matrix:
            out += matrix
            out.append("")
    for ds in datasets:
        grp = _sort_by_primary([rc for rc in records if rc["dataset"] == ds],
                               prim, specs, metric_cols)
        # 组内全空的列不显示（各数据集适用指标不同，如 TTS 的 GT 锚点只有部分集有）
        gcols = [c for c in cols if any(rc.get(c, "") not in ("", None) for rc in grp)]
        if not gcols:
            continue
        lang = grp[0].get("lang") or ""
        out.append(f"*{ds}*" + (f"（{lang}）" if lang else ""))
        out.append("")
        out += _render_rows(grp, gcols, specs)[0]
        out.append("")
    return "\n".join(out).rstrip()


def _fill_scope(module, cfg):
    tmpl = (cfg.get("modules", {}).get(module) or {}).get("scope", "")
    if not tmpl:
        return ""
    counts = dict((cfg.get("subset", {}).get(module) or {}))
    chip = cfg.get("models", {}).get("chip", "")
    try:
        s = tmpl.format(**counts)
    except (KeyError, IndexError):
        s = tmpl
    return f"{s} · {chip}" if chip else s


def build_status_matrix(rows, cfg):
    data_root = Path(cfg["paths"]["data_root"])
    models = cfg.get("models", {})
    tested = {r["module"] for r in rows}
    head = "| 模块 | 数据就绪 | 模型注册 | 指标已测 |\n|---|---|---|---|"
    lines = [head]
    for m in module_names(cfg):
        data_ok = "✅" if (data_root / "benchmark" / m).exists() else "—"
        model_ok = "✅" if models.get(m) else "—"
        test_ok = "✅" if m in tested else "—"
        lines.append(f"| {module_label(m, cfg)} | {data_ok} | {model_ok} | {test_ok} |")
    return "\n".join(lines)


def render_summary_md(rows, cfg):
    chip = cfg.get("models", {}).get("chip", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = ["# Voice_Test.AXERA 基准结果",
           "",
           f"> {chip} 芯片实测 · 共 {len(rows)} 条指标记录 · "
           f"由 `tools/report.py` 自动生成（勿手改）· {ts}",
           "",
           "## 模块就绪度",
           "",
           build_status_matrix(rows, cfg),
           ""]
    for m in module_names(cfg):
        mrows = [r for r in rows if r["module"] == m]
        if not mrows:
            continue
        out.append(f"## {module_label(m, cfg)}")
        out.append("")
        out.append(render_module_table(m, mrows, cfg))
        out.append("")
    return "\n".join(out)


def _replace_block(text, tag, content):
    """幂等替换 <!-- tag -->…<!-- /tag -->；无标记返回 (text, False)。"""
    a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    i, j = text.find(a), text.find(b)
    if i < 0 or j < 0 or j < i:
        return text, False
    return text[:i + len(a)] + "\n" + content + "\n" + text[j:], True


def inject_readme(readme, rows, cfg, dry_run=False):
    text = readme.read_text(encoding="utf-8")
    missing = []
    for m in module_names(cfg):
        mrows = [r for r in rows if r["module"] == m]
        content = render_module_table(m, mrows, cfg) if mrows else "_（暂无实测数据）_"
        text, ok = _replace_block(text, f"RESULTS:{m}", content)
        if not ok:
            missing.append(f"RESULTS:{m}")
    text, ok = _replace_block(text, "STATUS-MATRIX", build_status_matrix(rows, cfg))
    if not ok:
        missing.append("STATUS-MATRIX")
    if missing:
        print(f"  [warn] README 缺少标记区块：{missing}（该区块跳过）")
    if dry_run:
        print("  [dry-run] 未写文件；注入后 README 长度", len(text))
    else:
        readme.write_text(text, encoding="utf-8")
        print(f"  已注入 → {readme}")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true", help="注入顶层 README 标记区块")
    ap.add_argument("--only", default="", help="只渲染某模块（调试打印）")
    ap.add_argument("--dry-run", action="store_true", help="预览不写文件")
    args = ap.parse_args()
    cfg = load_config()
    results_dir = REPO_ROOT / cfg["paths"]["results_dir"]
    rows = load_rows(results_dir)
    if not rows:
        print(f"results/ 下无有效 csv（{results_dir}），先跑 run_benchmark.sh")
        return
    if args.only:
        print(render_module_table(args.only, [r for r in rows if r["module"] == args.only], cfg))
        return
    md = render_summary_md(rows, cfg)
    summary = results_dir / "summary.md"
    if args.dry_run:
        print(md)
    else:
        summary.write_text(md, encoding="utf-8")
        print(f"汇总 {len(rows)} 条 → {summary}")
    if args.readme:
        readme = REPO_ROOT / cfg.get("report", {}).get("readme", "README.md")
        inject_readme(readme, rows, cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
