#!/bin/bash
# ============================================================
# VAD 基准（板端 NPU 执行）
# 1) 精度：SileroVAD 在 benchmark VAD 子集上算帧级 F1/AUC/acc（eval_silero.py）
# 2) 性能：silero benchmark.py 测 RTF（整段+流式）、CMM/OS 内存
# 依赖：model_root/silero_vad_axera-0.1.2 已 pip install -e（提供 silero_vad_axera 包）
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
CHIP="$($PY "$REPO_DIR/tools/cfg.py" models.chip)"

# 精度评测
if $PY -c "import silero_vad_axera" 2>/dev/null; then
    echo ">>> VAD 精度（SileroVAD）"
    $PY "$REPO_DIR/vad/eval_silero.py" --backend "$CHIP" || echo "  eval_silero 失败"
else
    echo "  未装 silero_vad_axera 包：cd $MODEL_ROOT/silero_vad_axera-0.1.2 && pip install --no-deps -e ."
fi

# 性能基准（RTF/内存）
if [ -f "$MODEL_ROOT/silero_vad_axera-0.1.2/benchmark.py" ]; then
    echo ">>> VAD 性能（RTF/内存）"
    ( cd "$MODEL_ROOT/silero_vad_axera-0.1.2" && $PY benchmark.py ) | tee "$REPO_DIR/results/vad_perf.log"
fi
echo ">>> VAD 完成"
