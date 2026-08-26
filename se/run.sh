#!/bin/bash
# ============================================================
# SE 基准（板端 NPU 执行）
# 对 configs/benchmark.yaml 的 models.se 里每个模型：增强 benchmark noisy 子集 -> enhanced/，
# 再算 PESQ/STOI/SI-SNR（vs clean）+ RTF，写 results/se.csv。
# 同时给出 noisy-vs-clean 基线作增强增益参照。
# 手动指定：SE_MODELS="gtcrn fastenhancer" bash se/run.sh
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
CHIP="$($PY "$REPO_DIR/tools/cfg.py" models.chip)"

# 基线（noisy vs clean）
echo ">>> SE noisy 基线（未增强参照）"
$PY "$REPO_DIR/se/eval_se.py"

# 已配置的 SE 模型
MODELS="${SE_MODELS:-$($PY -c "import sys;sys.path.insert(0,'$REPO_DIR/tools');from common import load_config;print(' '.join(load_config()['models'].get('se',{}).keys()))")}"
for m in $MODELS; do
    echo ">>> SE 模型: $m"
    ENH_DIR="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)/benchmark/se/enhanced/$m"
    if $PY "$REPO_DIR/se/enhance_warm.py" --model "$m" --chip "$CHIP"; then
        RTF="$(cat "$ENH_DIR/_rtf.txt" 2>/dev/null || echo '')"
        $PY "$REPO_DIR/se/eval_se.py" --enhanced_dir "$ENH_DIR" --model_name "$m" --rtf "$RTF"
    else
        echo "  [$m] 增强失败（检查模型目录/axengine）"
    fi
done
echo ">>> SE 完成 -> results/se.csv"
