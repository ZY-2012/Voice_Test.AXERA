#!/bin/bash
# ============================================================
# ① 环境配置
# 安装数据制作与指标评估所需 Python 依赖 + aria2c 下载器。
#   host 端（数据准备机）：直接跑，装全部依赖
#   AX 板端（跑推理）：加 --board，torch 系列装 CPU 版（避免 libcudart 报错）
# 用法: bash setup_env.sh [--board]
# ============================================================
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
BOARD=0
[ "$1" = "--board" ] && BOARD=1

echo ">>> [1/3] pip 依赖"
$PY -m pip install -q -r "$REPO_DIR/requirements.txt"

if [ "$BOARD" = "1" ]; then
    echo ">>> [2/3] 板端 torch CPU 版"
    $PY -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu || \
        echo "  (torch CPU 安装失败，若板端已装可忽略)"
else
    echo ">>> [2/3] host 端跳过 torch（如需 funasr 生成 VAD 标签会自动用已装 torch）"
fi

echo ">>> [3/3] aria2c 下载器"
if ! command -v aria2c >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        conda install -y -c conda-forge aria2 || echo "  conda 装 aria2 失败，请手动: apt install aria2"
    else
        echo "  未找到 aria2c，请手动安装: apt install aria2 或 conda install -c conda-forge aria2"
    fi
fi

echo ">>> 环境就绪。下一步: bash download_datasets.sh"
