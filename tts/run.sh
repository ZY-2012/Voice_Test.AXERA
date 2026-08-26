#!/bin/bash
# ============================================================
# TTS 基准（板端 NPU 执行）
# 对 configs 的 models.tts 每个模型 × 各数据集：
#   1) synth_batch.py 合成 text 子集 -> syn_dir（记录 RTF）
#   2) 构建回环 ASR 评测目录（合成音 + 输入文本作参考）
#   3) 复用回环 ASR（默认 sensevoice）test_wer.py 得 CER/WER（=可懂度）
#   4) 写 results/tts.csv
# 手动：TTS_MODELS="kokoro melotts" TTS_DATASETS="aishell3 ljspeech" bash tts/run.sh
# cosyvoice2 较重（C++ LLM，需 tokenizer server），默认 limit 由 TTS_LIMIT 控制。
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DATA_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
CHIP="$($PY "$REPO_DIR/tools/cfg.py" models.chip)"
ASR_MODEL="$($PY "$REPO_DIR/tools/cfg.py" eval.tts.asr_model)"   # 默认 sensevoice
CSV="$REPO_DIR/results/tts.csv"
mkdir -p "$REPO_DIR/results"
LOGD="$REPO_DIR/results/logs"; mkdir -p "$LOGD"   # 回环 ASR 日志（不用 /tmp）
[ -f "$CSV" ] || echo "module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note" > "$CSV"

MODELS="${TTS_MODELS:-$($PY -c "import sys;sys.path.insert(0,'$REPO_DIR/tools');from common import load_config;print(' '.join(load_config()['models'].get('tts',{}).keys()))")}"
DATASETS="${TTS_DATASETS:-aishell3 ljspeech}"
grab_cer() { grep -oiE "(total )?(wer|cer)[: ]+[0-9.]+%?" "$1" | tail -1 | grep -oE "[0-9.]+"; }

for m in $MODELS; do
  for ds in $DATASETS; do
    lang=zh; [ "$ds" = ljspeech ] && lang=en
    echo ">>> TTS $m / $ds ($lang)"
    # TTS_LIMIT>0 对所有模型限条数（冒烟/重模型用）；cosyvoice2 无 TTS_LIMIT 时默认 20
    LIM=""
    if [ "${TTS_LIMIT:-0}" -gt 0 ]; then LIM="--limit $TTS_LIMIT";
    elif [ "$m" = cosyvoice2 ]; then LIM="--limit 20"; fi
    SYN="$DATA_ROOT/benchmark/tts/syn/${m}_${ds}"
    $PY "$REPO_DIR/tts/synth_batch.py" --model "$m" --dataset "$ds" --chip "$CHIP" $LIM || { echo "  合成失败"; continue; }
    [ -d "$SYN" ] || continue
    # 构建回环 ASR 评测目录：ground_truth.txt(uttid 文本) + wav/(合成音软链)
    LB="$DATA_ROOT/benchmark/tts/loopback/${m}_${ds}"
    mkdir -p "$LB/wav"
    ln -sfn wav "$LB/aishell_S0764"   # SenseVoice/Whisper/WeNet 用 aishell_S0764 目录名
    : > "$LB/ground_truth.txt"
    while IFS=$'\t' read -r uid txt; do
        [ -f "$SYN/$uid.wav" ] || continue
        ln -sf "$SYN/$uid.wav" "$LB/wav/$uid.wav"
        echo "$uid $txt" >> "$LB/ground_truth.txt"
    done < "$DATA_ROOT/benchmark/tts/$ds/text"
    # 回环 ASR（sensevoice）
    RTF="$(cat "$SYN/_rtf.txt" 2>/dev/null || echo '')"
    if [ "$ASR_MODEL" = sensevoice ] && [ -d "$MODEL_ROOT/SenseVoice/python" ]; then
        ( cd "$MODEL_ROOT/SenseVoice/python" && \
          $PY test_wer.py -d aishell -g "$LB/ground_truth.txt" --language "$lang" >$LOGD/tts_lb.log 2>&1 )
        v=$(grab_cer $LOGD/tts_lb.log)
        met=$([ "$lang" = zh ] && echo cer_loopback || echo wer_loopback)
        echo "tts,$m,$ds,$lang,$met,${v:-NA},$RTF,,,回环ASR=$ASR_MODEL" >> "$CSV"
        echo "  回环 ${met}=${v:-NA}%  RTF=$RTF"
    else
        echo "  回环 ASR $ASR_MODEL 未就绪，仅合成完成（syn=$SYN, RTF=$RTF）"
        echo "tts,$m,$ds,$lang,rtf,$RTF,$RTF,,,合成完成待回环" >> "$CSV"
    fi
  done
done
echo ">>> TTS 完成 -> $CSV"
