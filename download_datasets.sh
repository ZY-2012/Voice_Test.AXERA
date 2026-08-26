#!/bin/bash
# ============================================================
# ② 下载原始数据
# 把四个模块所需原始数据下载到 data_root（configs/benchmark.yaml）。
# 幂等：已存在的目标自动跳过，可反复运行续传。
# 用法:
#   bash download_datasets.sh              # 全部模块
#   bash download_datasets.sh asr vad      # 指定模块（asr/vad/se/tts）
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
DATA_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)"
MIRROR="$($PY "$REPO_DIR/tools/cfg.py" download.hf_endpoint)"
export HF_ENDPOINT="$MIRROR"
CONN=16
OPTS="-x$CONN -s$CONN -k1M -c --auto-file-renaming=false --console-log-level=warn"
MODULES=("${@:-asr vad se tts}")
[ $# -gt 0 ] && MODULES=("$@") || MODULES=(asr vad se tts)

has() { for m in "${MODULES[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }
need_aria() { command -v aria2c >/dev/null || { echo "!!! 需要 aria2c，先跑 setup_env.sh"; exit 1; }; }

# ---------------- ASR ----------------
dl_asr() {
    echo "==== [ASR] 英文 ESB + 中文 AISHELL-1 ===="
    need_aria
    local M="$MIRROR/datasets"
    # 英文 ESB（test-only）
    mkdir -p "$DATA_ROOT/asr/librispeech" "$DATA_ROOT/asr/voxpopuli" "$DATA_ROOT/asr/tedlium" \
             "$DATA_ROOT/asr/earnings22/chunked" "$DATA_ROOT/asr/ami/audio" "$DATA_ROOT/asr/ami/annotations"
    [ -f "$DATA_ROOT/asr/librispeech/test-clean.tar.gz" ] || aria2c $OPTS -d "$DATA_ROOT/asr/librispeech" \
        "http://www.openslr.org/resources/12/test-clean.tar.gz" "http://www.openslr.org/resources/12/test-other.tar.gz"
    [ -f "$DATA_ROOT/asr/voxpopuli/test-00000-of-00001.parquet" ] || aria2c $OPTS -d "$DATA_ROOT/asr/voxpopuli" \
        -o test-00000-of-00001.parquet "$M/polinaeterna/voxpopuli/resolve/main/en/test-00000-of-00001.parquet"
    [ -f "$DATA_ROOT/asr/tedlium/test-00000-of-00001.parquet" ] || aria2c $OPTS -d "$DATA_ROOT/asr/tedlium" \
        -o test-00000-of-00001.parquet "$M/AudioLLMs/tedlium3_test/resolve/main/data/test-00000-of-00001.parquet"
    [ -f "$DATA_ROOT/asr/earnings22/metadata.csv" ] || aria2c $OPTS -d "$DATA_ROOT/asr/earnings22" \
        -o metadata.csv "$M/anton-l/earnings22_baseline_5_gram/resolve/main/metadata.csv"
    for id in 4432298 4450488 4470290 4479741 4483338 4485244; do
        [ -f "$DATA_ROOT/asr/earnings22/chunked/$id.tar.gz" ] || aria2c $OPTS -d "$DATA_ROOT/asr/earnings22/chunked" \
            -o "$id.tar.gz" "$M/anton-l/earnings22_baseline_5_gram/resolve/main/data/chunked/$id.tar.gz"
    done
    for id in EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d \
              IS1009a IS1009b IS1009c IS1009d TS3003a TS3003b TS3003c TS3003d; do
        [ -f "$DATA_ROOT/asr/ami/audio/$id.tar.gz" ] || aria2c $OPTS -d "$DATA_ROOT/asr/ami/audio" \
            -o "$id.tar.gz" "$M/speech-seq2seq/ami/resolve/main/audio/ihm/eval/$id.tar.gz"
    done
    [ -f "$DATA_ROOT/asr/ami/annotations/text" ] || aria2c $OPTS -d "$DATA_ROOT/asr/ami/annotations" \
        -o text "$M/speech-seq2seq/ami/resolve/main/annotations/eval/text"
    # gigaspeech / spgispeech 为 gated（需 HF 同意条款+token），可选，见 README
    # 中文 AISHELL-1
    if [ ! -d "$DATA_ROOT/asr/aishell1/wav" ]; then
        echo "-- AISHELL-1（modelscope）--"
        $PY -c "from modelscope import MsDataset; MsDataset.load('speech_asr/speech_asr_aishell1_testsets')" 2>/dev/null || \
        $PY -m modelscope.cli.cli download --dataset modelscope/speech_asr_aishell1_testsets \
            --local_dir "$DATA_ROOT/vad/aishell1" 2>/dev/null || \
        echo "  (AISHELL-1 请手动: modelscope download --dataset modelscope/speech_asr_aishell1_testsets --local_dir $DATA_ROOT/vad/aishell1)"
    fi
}

# ---------------- VAD ----------------
dl_vad() {
    echo "==== [VAD] LibriVAD 对齐/噪声 + DEMAND 噪声 ===="
    need_aria
    local M="$MIRROR/datasets"
    mkdir -p "$DATA_ROOT/vad/librivad"
    [ -f "$DATA_ROOT/vad/librivad/Forced_alignments.zip" ] || aria2c $OPTS -d "$DATA_ROOT/vad/librivad" \
        -o Forced_alignments.zip "$M/LibriVAD/LibriVAD/resolve/main/Files/Forced_alignments.zip"
    [ -f "$DATA_ROOT/vad/librivad/Noises.zip" ] || aria2c $OPTS -d "$DATA_ROOT/vad/librivad" \
        -o Noises.zip "$M/LibriVAD/LibriVAD/resolve/main/Files/Noises.zip"
    # DEMAND 噪声（Picovoice VAD 混噪用）：Kaggle，需 kaggle token（~/.kaggle/kaggle.json）
    if [ ! -f "$DATA_ROOT/vad/demand/demand_archive.zip" ] && [ ! -d "$DATA_ROOT/vad/demand/extracted" ]; then
        mkdir -p "$DATA_ROOT/vad/demand"
        if command -v kaggle >/dev/null 2>&1; then
            echo "-- DEMAND（kaggle）--"
            kaggle datasets download -d aanhari/demand-dataset -p "$DATA_ROOT/vad/demand" || \
            echo "  kaggle 下载失败，请手动下载 https://www.kaggle.com/datasets/aanhari/demand-dataset 到 $DATA_ROOT/vad/demand/"
        else
            echo "  !! DEMAND 需 Kaggle：pip install kaggle 并配置 ~/.kaggle/kaggle.json，或手动下载"
            echo "     https://www.kaggle.com/datasets/aanhari/demand-dataset  ->  $DATA_ROOT/vad/demand/demand_archive.zip"
        fi
    fi
    echo "  (LibriSpeech test-clean 语音复用 asr 模块，无需重复下载)"
}

# ---------------- SE ----------------
dl_se() {
    echo "==== [SE] VoiceBank-DEMAND (test) ===="
    need_aria
    mkdir -p "$DATA_ROOT/se/voicebank_demand"
    [ -f "$DATA_ROOT/se/voicebank_demand/test-00000-of-00001.parquet" ] || aria2c $OPTS \
        -d "$DATA_ROOT/se/voicebank_demand" -o test-00000-of-00001.parquet \
        "$MIRROR/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main/data/test-00000-of-00001.parquet"
}

# ---------------- TTS ----------------
dl_tts() {
    echo "==== [TTS] LJSpeech(en test) + AISHELL-3(zh test) ===="
    need_aria
    local M="$MIRROR/datasets"
    # LJSpeech：按 test.txt 清单只下 test wav
    mkdir -p "$DATA_ROOT/tts/ljspeech/wav"
    [ -f "$DATA_ROOT/tts/ljspeech/test.txt" ] || curl -sL "$M/flexthink/ljspeech/raw/main/test.txt" -o "$DATA_ROOT/tts/ljspeech/test.txt"
    [ -f "$DATA_ROOT/tts/ljspeech/metadata.csv" ] || curl -sL "$M/flexthink/ljspeech/resolve/main/metadata.csv" -o "$DATA_ROOT/tts/ljspeech/metadata.csv"
    if [ "$(ls "$DATA_ROOT/tts/ljspeech/wav" 2>/dev/null | wc -l)" -lt 400 ]; then
        awk '{print "'$M'/flexthink/ljspeech/resolve/main/wavs/"$1".wav\n  out="$1".wav"}' \
            "$DATA_ROOT/tts/ljspeech/test.txt" > "$DATA_ROOT/tts/ljspeech/urls.txt"
        aria2c -x4 -s4 -c --auto-file-renaming=false --console-log-level=warn \
            -d "$DATA_ROOT/tts/ljspeech/wav" -i "$DATA_ROOT/tts/ljspeech/urls.txt"
    fi
    # AISHELL-3 test（HF 子集，44.1kHz）
    if [ ! -d "$DATA_ROOT/tts/aishell3/test/wav" ]; then
        $PY -c "
import os; os.environ['HF_ENDPOINT']='$MIRROR'
from huggingface_hub import snapshot_download
snapshot_download('AISHELL/AISHELL-3', repo_type='dataset',
    allow_patterns=['test/*','phone_set.txt','spk-info.txt'],
    local_dir='$DATA_ROOT/tts/aishell3')
print('AISHELL-3 test 下载完成')
"
    fi
}

for m in "${MODULES[@]}"; do
    case "$m" in
        asr) dl_asr ;;
        vad) dl_vad ;;
        se)  dl_se ;;
        tts) dl_tts ;;
        *) echo "未知模块: $m（可选 asr/vad/se/tts）" ;;
    esac
done
echo ">>> 下载完成。下一步: bash prepare_datasets.sh"
