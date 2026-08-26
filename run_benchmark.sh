#!/bin/bash
# ============================================================
# ⑤ 跑基准测试（板端 NPU）
# 依次调用各模块 run.sh，产出指标到 results/，并汇总成 results/summary.md。
# 需在 AX 板端 base 环境执行（axengine + 各模型推理代码就绪）。
# 用法:
#   bash run_benchmark.sh                 # 全部模块
#   bash run_benchmark.sh asr vad         # 指定模块
#   DATASET=aishell1 bash run_benchmark.sh asr   # 指定单数据集
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
RESULTS="$REPO_DIR/$($PY "$REPO_DIR/tools/cfg.py" paths.results_dir)"
mkdir -p "$RESULTS"
[ $# -gt 0 ] && MODULES=("$@") || MODULES=(asr vad se tts)

for m in "${MODULES[@]}"; do
    if [ -x "$REPO_DIR/$m/run.sh" ] || [ -f "$REPO_DIR/$m/run.sh" ]; then
        echo "############## 模块 $m ##############"
        bash "$REPO_DIR/$m/run.sh" || echo "  [$m] run.sh 返回非零（部分模型可能未就绪）"
    else
        echo "  [$m] 无 run.sh，跳过"
    fi
done

echo "############## 汇总 ##############"
$PY "$REPO_DIR/tools/aggregate_results.py" && \
echo ">>> 结果: $RESULTS/summary.md"
