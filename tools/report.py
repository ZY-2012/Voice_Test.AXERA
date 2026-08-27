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
    lower_better: bool   # 越低越好


# 指标 → 显示规则；未列出的兜底 3 位小数、越高越好
METRIC_FORMATS = {
    "cer":           Fmt("CER", 2, "%", True),
    "wer":           Fmt("WER", 2, "%", True),
    "cer_loopback":  Fmt("回环CER", 2, "%", True),
    "wer_loopback":  Fmt("回环WER", 2, "%", True),
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
    "cmm_mb":        Fmt("CMM(MB)", -1, "", True),
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
    每条 record: {model, dataset, <metric>: value..., rtf, cmm_mb, os_mb, note}"""
    order = []          # (model,dataset) 保持首见顺序
    rec = {}
    metric_cols = []
    for r in module_rows:
        key = (r["model"], r["dataset"])
        if key not in rec:
            rec[key] = {"model": r["model"], "dataset": r["dataset"],
                        "rtf": r.get("rtf", ""), "cmm_mb": r.get("cmm_mb", ""),
                        "os_mb": r.get("os_mb", ""), "note": r.get("note", "")}
            order.append(key)
        m = r["metric"]
        if m not in metric_cols:
            metric_cols.append(m)
        rec[key][m] = r.get("value", "")
        # rtf/内存取该模型首个非空
        for c in ("rtf", "cmm_mb", "os_mb"):
            if not rec[key].get(c) and r.get(c):
                rec[key][c] = r[c]
    records = [rec[k] for k in order]
    has_multi_ds = len({d for _, d in order}) > 1
    extra = [c for c in ("rtf", "cmm_mb", "os_mb")
             if any(rec[k].get(c) for k in order)]
    return records, metric_cols, has_multi_ds, extra


def _best_index(records, col, spec):
    """该列最优行索引（≥2 行才返回，否则 None）。"""
    vals = []
    for i, rc in enumerate(records):
        try:
            vals.append((i, float(rc.get(col, ""))))
        except (TypeError, ValueError):
            pass
    if len(vals) < 2:
        return None
    return (min if spec.lower_better else max)(vals, key=lambda t: t[1])[0]


def render_module_table(module, module_rows, cfg):
    records, metric_cols, has_multi_ds, extra = pivot(module_rows)
    if not records:
        return ""
    cols = metric_cols + extra                      # 指标列 + rtf/内存
    specs = {c: METRIC_FORMATS.get(c, _DEFAULT_FMT) for c in cols}
    # 排序：按主指标
    prim = (cfg.get("report", {}).get("primary_metric", {}) or {}).get(module)
    if prim in metric_cols:
        ps = specs[prim]
        def _k(rc):
            try:
                return (0, float(rc.get(prim, "")) * (1 if ps.lower_better else -1))
            except (TypeError, ValueError):
                return (1, 0.0)
        records = sorted(records, key=_k)
    best = {c: _best_index(records, c, specs[c]) for c in cols}

    head = (["模型"] + (["数据集"] if has_multi_ds else [])
            + [specs[c].label or c for c in cols])
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for i, rc in enumerate(records):
        cells = [rc["model"]] + ([rc["dataset"]] if has_multi_ds else [])
        for c in cols:
            txt = fmt_value(rc.get(c, ""), specs[c])
            if best[c] == i and txt != "—":
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")

    # 口径标注 + 小结
    scope = _fill_scope(module, cfg)
    out = [f"**{module_label(module, cfg)}**"]
    if scope:
        out.append(f"> 口径：{scope}")
    out.append("")
    out += lines
    if prim in metric_cols and best.get(prim) is not None:
        bi = best[prim]
        out.append("")
        out.append(f"> 最优：{records[bi]['model']} "
                   f"{specs[prim].label}={fmt_value(records[bi].get(prim, ''), specs[prim])}")
    return "\n".join(out)


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
