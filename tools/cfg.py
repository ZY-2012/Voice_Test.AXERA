#!/usr/bin/env python3
"""从 configs/benchmark.yaml 按点分 key 取值，供 sh 脚本调用。
用法: python tools/cfg.py paths.data_root
"""
import sys
from common import load_config

if __name__ == "__main__":
    cfg = load_config()
    node = cfg
    for k in sys.argv[1].split("."):
        node = node[k]
    print(node)
