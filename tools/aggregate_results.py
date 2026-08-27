#!/usr/bin/env python3
"""[deprecated 兼容入口] 汇总 results/*.csv → results/summary.md。
渲染逻辑已迁移到 tools/report.py（美化透视 + README 注入），请直接使用：
  python tools/report.py            # 生成 summary.md
  python tools/report.py --readme   # 额外注入顶层 README
保留本文件仅为兼容旧调用点（如 run_benchmark.sh 历史版本）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report

if __name__ == "__main__":
    report.main()
