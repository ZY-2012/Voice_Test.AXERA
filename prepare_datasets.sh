#!/bin/bash
# ============================================================
# ③ 制作测试集
# 解压/转 16k wav/生成标签/整理清单，最后确定性抽取标准子集到 benchmark/。
# 幂等：可重复运行。VAD 中文标签需 funasr（GPU 更快），无 GPU 会很慢，可跳过。
# 用法:
#   bash prepare_datasets.sh                 # 全部
#   bash prepare_datasets.sh asr se          # 指定模块
#   FULL=1 bash prepare_datasets.sh          # 子集用全量
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
DATA_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)"
[ $# -gt 0 ] && MODULES=("$@") || MODULES=(asr vad se tts)
FULL_FLAG=""; [ "${FULL:-0}" = "1" ] && FULL_FLAG="--full"
has() { for m in "${MODULES[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

if has asr; then
    echo "==== [ASR] 制作 ===="
    $PY "$REPO_DIR/asr/prepare_asr_test.py" --base "$DATA_ROOT/asr" || true   # ESB 英文
    $PY "$REPO_DIR/asr/prepare_aishell.py"                                    # 中文 AISHELL-1
fi

if has vad; then
    echo "==== [VAD] 制作 ===="
    # DEMAND 解压（Picovoice 混噪用）
    if [ -f "$DATA_ROOT/vad/demand/demand_archive.zip" ] && [ ! -d "$DATA_ROOT/vad/demand/extracted" ]; then
        $PY -c "import zipfile; zipfile.ZipFile('$DATA_ROOT/vad/demand/demand_archive.zip').extractall('$DATA_ROOT/vad/demand/extracted')"
    fi
    # LibriVAD 对齐解压（test-clean）
    if [ -f "$DATA_ROOT/vad/librivad/Forced_alignments.zip" ] && [ ! -d "$DATA_ROOT/vad/librivad/alignments" ]; then
        $PY -c "
import zipfile
z=zipfile.ZipFile('$DATA_ROOT/vad/librivad/Forced_alignments.zip')
[z.extract(n,'$DATA_ROOT/vad/librivad/alignments') for n in z.namelist() if '/test-clean/' in n and n.endswith('.TextGrid')]
print('LibriVAD test-clean 对齐已解压')"
    fi
    # LibriVAD 采样级标签（依赖 asr 的 test-clean wav + praat-textgrids）
    [ -d "$DATA_ROOT/asr/librispeech/wav" ] && $PY "$REPO_DIR/vad/gen_librivad_test.py" || echo "  跳过 LibriVAD（需先制作 asr）"
    # Picovoice 长流（需 DEMAND 解压 + test-clean wav）
    [ -d "$DATA_ROOT/vad/demand/extracted" ] && $PY "$REPO_DIR/vad/gen_picovoice_test.py" || echo "  跳过 Picovoice（需 DEMAND）"
    # 中文 VAD 标签（funasr，GPU 快；无 GPU 可加 --device cpu 但很慢）
    if [ -d "$DATA_ROOT/asr/aishell1/wav" ]; then
        $PY "$REPO_DIR/vad/gen_aishell_vad_labels.py" --split test --device "${VAD_DEVICE:-cuda:0}" || \
        echo "  中文 VAD 标签生成失败（检查 funasr/设备），可稍后重试"
    fi
fi

if has se; then
    echo "==== [SE] 制作 ===="
    $PY "$REPO_DIR/se/prepare_se.py"
fi

if has tts; then
    echo "==== [TTS] 制作 ===="
    $PY "$REPO_DIR/tts/prepare_tts.py"
fi

echo "==== 抽取标准子集 -> $DATA_ROOT/benchmark ===="
$PY "$REPO_DIR/tools/make_subsets.py" --module "${MODULES[@]}" $FULL_FLAG

echo ">>> 制作完成。下一步: bash download_models.sh && bash run_benchmark.sh"
