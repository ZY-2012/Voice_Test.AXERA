#!/usr/bin/env python3
"""汇总 results/*.csv -> results/summary.md（Markdown 指标总表）。
统一 CSV 表头: module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note
各模块 run.sh 追加行，末尾运行本脚本生成总表。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config

HEADER = ["module", "model", "dataset", "lang", "metric", "value",
          "rtf", "cmm_mb", "os_mb", "note"]


def main():
    cfg = load_config()
    results = Path(__file__).resolve().parent.parent / cfg["paths"]["results_dir"]
    rows = []
    for csvf in sorted(results.glob("*.csv")):
        with open(csvf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    if not rows:
        print("results/ 下暂无 csv，先跑 run_benchmark.sh")
        return

    md = ["# Voice_Test.AXERA 基准结果", "",
          f"> 共 {len(rows)} 条记录，来自 {cfg['models']['chip']} 芯片实测。", ""]
    for module in ["asr", "vad", "se", "tts"]:
        mrows = [r for r in rows if r["module"] == module]
        if not mrows:
            continue
        md.append(f"## {module.upper()}")
        md.append("")
        md.append("| 模型 | 数据集 | 语言 | 指标 | 数值 | RTF | CMM(MB) | OS(MB) | 备注 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in mrows:
            md.append("| {model} | {dataset} | {lang} | {metric} | {value} | "
                      "{rtf} | {cmm_mb} | {os_mb} | {note} |".format(**{k: r.get(k, "") for k in HEADER}))
        md.append("")
    out = results / "summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"汇总 {len(rows)} 条 -> {out}")


if __name__ == "__main__":
    main()
