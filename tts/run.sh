#!/bin/bash
# ============================================================
# TTS 基准（板端 NPU 执行）
# 对 configs 的 models.tts 每个模型 × 各数据集：
#   1) synth_batch.py 合成 text 子集 -> syn_dir（记录 RTF）
#   2) GT 锚点（每数据集一次，幂等）：参考真人音频过同一 ASR -> ASR 地板
#   3) asr_transcribe.py 转写合成音 -> hyp.txt
#   4) eval_tts.py 算分：回环 CER/WER + S/D/I 分解 + len_ratio + success_rate
#   5) 写 results/tts.csv
#
# 与旧版差异：不再调 vendor test_wer.py 抓 log 数字。原因见 tts/README.md §4.2
#   - vendor 用 line.split(" ") 解析 gt，英文含空格必崩（英文回环从来跑不出结果）
#   - vendor 的 "WER" 是字符级；英文需词级，现由 tools/metrics_asr.py 统一口径
#
# 手动：TTS_MODELS="kokoro melotts" TTS_DATASETS="aishell3 ljspeech" bash tts/run.sh
#       TTS_LIMIT=3 bash tts/run.sh          # 冒烟
#       TTS_SKIP_ANCHOR=1 bash tts/run.sh    # 跳过 GT 锚点（已测过）
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DATA_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)"
CHIP="$($PY "$REPO_DIR/tools/cfg.py" models.chip)"
CSV="$REPO_DIR/results/tts.csv"
mkdir -p "$REPO_DIR/results"
LOGD="$REPO_DIR/results/logs"; mkdir -p "$LOGD"
[ -f "$CSV" ] || echo "module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note" > "$CSV"

MODELS="${TTS_MODELS:-$($PY -c "import sys;sys.path.insert(0,'$REPO_DIR/tools');from common import load_config;print(' '.join(k for k,v in load_config()['models'].get('tts',{}).items() if (v or {}).get('enabled', True) is not False))")}"
DATASETS="${TTS_DATASETS:-aishell3 ljspeech}"

# 分语言选 ASR：中文 firered（CER 地板最低），英文待 GT 锚点定夺
asr_for_lang() {
    if [ "$1" = zh ]; then
        $PY "$REPO_DIR/tools/cfg.py" eval.tts.asr_model_zh 2>/dev/null || echo firered
    else
        $PY "$REPO_DIR/tools/cfg.py" eval.tts.asr_model_en 2>/dev/null || echo sensevoice
    fi
}

for ds in $DATASETS; do
  # 语言从 tools/common.py 的 TTS_DATASETS 派生（单一事实来源，勿在此另写 case 分支）
  lang="$($PY -c "import sys;sys.path.insert(0,'$REPO_DIR/tools');from common import tts_lang;print(tts_lang('$ds'))" 2>/dev/null)"
  [ -n "$lang" ] || { echo ">>> 未知数据集 $ds（需在 tools/common.py 的 TTS_DATASETS 登记），跳过"; continue; }
  ASR="$(asr_for_lang $lang)"
  BENCH="$DATA_ROOT/benchmark/tts/$ds"
  [ -f "$BENCH/text" ] || { echo ">>> 无子集 $BENCH/text，跳过 $ds"; continue; }
  LIM=""; [ "${TTS_LIMIT:-0}" -gt 0 ] && LIM="--limit $TTS_LIMIT"

  # ---- GT 锚点：参考真人音频过同一 ASR，得 ASR 地板（每数据集一次）----
  # 必须先跑：回环值须扣掉锚点才能解读（区分"TTS 差"与"ASR 差/归一化 bug"）
  if [ "${TTS_SKIP_ANCHOR:-0}" != 1 ] && [ -f "$BENCH/wav.scp" ]; then
    if grep -q "^tts,GT,$ds," "$CSV" 2>/dev/null; then
      echo ">>> GT 锚点 $ds 已存在，跳过（TTS_SKIP_ANCHOR=0 且删除 CSV 中 GT 行可重测）"
    else
      echo ">>> GT 锚点 $ds ($lang, ASR=$ASR)"
      GT_HYP="$LOGD/gt_${ds}_${ASR}.hyp"
      if $PY "$REPO_DIR/tts/asr_transcribe.py" --asr "$ASR" --scp "$BENCH/wav.scp" \
             --out "$GT_HYP" --lang "$lang" --chip "$CHIP" --ids "$BENCH/text" $LIM; then
        $PY "$REPO_DIR/tts/eval_tts.py" --dataset "$ds" --gt_anchor \
            --asr_hyp "$GT_HYP" --asr_name "$ASR" $LIM
      else
        echo "  !! GT 锚点转写失败（回环值将无法扣除 ASR 地板），继续跑模型"
      fi
    fi
  fi

  # ---- 逐模型：合成 -> 转写 -> 算分 ----
  for m in $MODELS; do
    echo ">>> TTS $m / $ds ($lang, ASR=$ASR)"
    MLIM="$LIM"
    # cosyvoice2 较重（C++ LLM，需 tokenizer server），无显式 TTS_LIMIT 时默认 20
    [ -z "$MLIM" ] && [ "$m" = cosyvoice2 ] && MLIM="--limit 20"
    SYN="$DATA_ROOT/benchmark/tts/syn/${m}_${ds}"
    # 有 batch_driver 的模型（如 melotts）用"载入一次+warmup"驱动合成 —— 这是纯 RTF 的正确来源；
    # 逐条起子进程的 synth_batch 每条都含模型加载，只能得到 e2e RTF
    DRV="$($PY -c "import sys;sys.path.insert(0,'$REPO_DIR/tools');from model_registry import REGISTRY;print(REGISTRY.get('$m',{}).get('batch_driver',''))" 2>/dev/null)"
    if [ -n "$DRV" ] && [ -f "$REPO_DIR/tts/$DRV" ]; then
      echo "  合成用批量驱动 $DRV（载入一次，纯 RTF）"
      $PY "$REPO_DIR/tts/$DRV" --dataset "$ds" $MLIM || { echo "  合成失败，跳过"; continue; }
    else
      # TTS_FRESH=1：忽略已存在 wav 强制重合成（合成口径/参数变更后作废旧产物时用）
      $PY "$REPO_DIR/tts/synth_batch.py" --model "$m" --dataset "$ds" --chip "$CHIP" $MLIM \
          ${TTS_FRESH:+--fresh} || { echo "  合成失败，跳过"; continue; }
    fi
    [ -d "$SYN" ] || continue
    RTF_PURE="$(cat "$SYN/_rtf_pure.txt" 2>/dev/null || echo '')"
    RTF="$(cat "$SYN/_rtf.txt" 2>/dev/null || echo '')"

    HYP="$LOGD/${m}_${ds}_${ASR}.hyp"
    # --ids + --limit：按 text 行序取前 N，与 synth_batch 的 rows[:N] 严格同口径
    $PY "$REPO_DIR/tts/asr_transcribe.py" --asr "$ASR" --wav_dir "$SYN" \
        --out "$HYP" --lang "$lang" --chip "$CHIP" --ids "$BENCH/text" $MLIM \
        || { echo "  转写失败，跳过"; continue; }
    $PY "$REPO_DIR/tts/eval_tts.py" --dataset "$ds" --syn_dir "$SYN" \
        --asr_hyp "$HYP" --model_name "$m" --asr_name "$ASR" $MLIM

    # RTF 两个口径分列写入：
    #   rtf     = 主口径，**不含模型加载**（批量驱动的热启动值 或 二进制自报推理计时）
    #   rtf_e2e = synth_batch 逐条子进程墙钟，含每条的模型加载（仅作参考，不可与主口径混比）
    # 空值/0/nan 一律不写：幂等重跑全部命中跳过时会算出 0，写入会污染已有指标。
    case "$RTF_PURE" in
      ""|"0.0000"|"0"|*nan*) [ -n "$RTF_PURE" ] && echo "  纯 RTF=$RTF_PURE 无效，不写入" ;;
      *) echo "tts,$m,$ds,$lang,rtf,$RTF_PURE,$RTF_PURE,,,不含模型加载(主口径);RTF随音频均长变化见audio_avg_s" >> "$CSV" ;;
    esac
    case "$RTF" in
      ""|"0.0000"|"0"|*nan*) [ -n "$RTF" ] && echo "  RTF(e2e)=$RTF 无效（未重新合成），不写入 CSV" ;;
      *) echo "tts,$m,$ds,$lang,rtf_e2e,$RTF,,,,synth_batch逐条子进程;含模型加载,非主口径" >> "$CSV" ;;
    esac
  done
done
echo ">>> TTS 完成 -> $CSV"
echo ">>> 刷新报表：python tools/report.py --readme"
