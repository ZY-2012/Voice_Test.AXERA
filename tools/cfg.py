#!/usr/bin/env python3
"""从 configs/benchmark.yaml 按点分 key 取值，供 sh 脚本调用。
用法:
  python tools/cfg.py paths.data_root      # 标量 → 原值
  python tools/cfg.py modules              # dict → 空格分隔的键（模块清单）
  python tools/cfg.py models.asr           # dict → 空格分隔的键（模型名）
"""
import sys
from common import load_config

if __name__ == "__main__":
    cfg = load_config()
    node = cfg
    for k in sys.argv[1].split("."):
        node = node[k]
    # dict → 键（空格分隔，供 shell 遍历）；list → 元素；标量 → 原值
    if isinstance(node, dict):
        print(" ".join(node.keys()))
    elif isinstance(node, (list, tuple)):
        print(" ".join(str(x) for x in node))
    else:
        print(node)
