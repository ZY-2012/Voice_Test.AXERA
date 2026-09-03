#!/bin/bash
# ============================================================
# VAD 基准（板端 NPU 执行）
# 模型清单来自 config models.vad，可用 VAD_MODELS='silero tenvad' 覆盖：
#   silero — python 包 silero_vad_axera，流式 chunk API（32ms 帧）
#   tenvad — C++ 可执行 ten_vad_example（16ms 帧），需先在 host 端 bash vad/build_tenvad.sh
# 1) 精度：benchmark VAD 子集上算帧级 F1/AUC/acc → results/vad.csv
# 2) 性能：RTF（不含模型加载）+ CMM/OS 内存
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
CHIP="$($PY "$REPO_DIR/tools/cfg.py" models.chip)"
MODELS="${VAD_MODELS:-$($PY "$REPO_DIR/tools/cfg.py" models.vad)}"

has_model() { for m in $MODELS; do [ "$m" = "$1" ] && return 0; done; return 1; }

# ---------------- SileroVAD（python 包） ----------------
if has_model silero; then
    if $PY -c "import silero_vad_axera" 2>/dev/null; then
        echo ">>> VAD 精度（SileroVAD）"
        $PY "$REPO_DIR/vad/eval_silero.py" --backend "$CHIP" || echo "  eval_silero 失败"
        # vendor 自带性能脚本（整段+流式 RTF、内存）
        if [ -f "$MODEL_ROOT/silero_vad_axera-0.1.2/benchmark.py" ]; then
            echo ">>> VAD 性能（SileroVAD RTF/内存）"
            ( cd "$MODEL_ROOT/silero_vad_axera-0.1.2" && $PY benchmark.py ) \
                | tee "$REPO_DIR/results/vad_perf.log"
        fi
    else
        echo "  未装 silero_vad_axera 包：cd $MODEL_ROOT/silero_vad_axera-0.1.2 && pip install --no-deps -e ."
    fi
fi

# ---------------- TEN VAD（C++ 可执行） ----------------
if has_model tenvad; then
    CODE_DIR="$MODEL_ROOT/$($PY "$REPO_DIR/tools/cfg.py" models.vad.tenvad.code_dir)"
    if [ -x "$CODE_DIR/install/example/ten_vad_example" ]; then
        echo ">>> VAD 精度+性能（TEN VAD）"
        $PY "$REPO_DIR/vad/eval_tenvad.py" || echo "  eval_tenvad 失败"
    else
        echo "  TEN VAD 未编译：host 端跑 bash vad/build_tenvad.sh（需 AX650 BSP SDK + aarch64 交叉编译器）"
    fi
fi

echo ">>> VAD 完成"
