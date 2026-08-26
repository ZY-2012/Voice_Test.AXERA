#!/bin/bash
# ============================================================
# ASR 基准（板端 NPU 执行）
# 对已适配 ASR 模型在 benchmark 子集上测 CER/WER，写 results/asr.csv。
# 默认用中文 aishell1 子集（与历史指标口径一致）；EN=1 追加英文子集。
# 依赖：各模型推理仓库在 model_root/<model>/（含各自 test_wer.py）。
# 口径：字符级 CER(zh)/词级 WER(en)，去标点转小写（tools/metrics_asr.py）。
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DATA_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.data_root)"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
BENCH="$DATA_ROOT/benchmark/asr"
CSV="$REPO_DIR/results/asr.csv"
mkdir -p "$REPO_DIR/results"
LOGD="$REPO_DIR/results/logs"; mkdir -p "$LOGD"   # 模型日志（不用 /tmp）
[ -f "$CSV" ] || echo "module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note" > "$CSV"

DATASET="${DATASET:-aishell1}"
GT="$BENCH/$DATASET/ground_truth.txt"
WAV="$BENCH/$DATASET/wav"
LANG=zh; [ "$DATASET" = "aishell1" ] || LANG=en
if [ ! -f "$GT" ]; then echo "无子集 $GT，先跑 prepare_datasets.sh"; exit 1; fi

# LIMIT>0 时截断 ground_truth（冒烟用；wav/ 目录共用，多余条目不影响）
if [ "${LIMIT:-0}" -gt 0 ]; then
    GT_LIM="$BENCH/$DATASET/ground_truth.lim.txt"
    head -n "$LIMIT" "$GT" > "$GT_LIM"; GT="$GT_LIM"
fi

# 从模型日志抓 CER/WER 百分数（各仓库 test_wer.py 打印 "Total WER: xx%"）
grab_cer() { grep -oiE "(total )?(wer|cer)[: ]+[0-9.]+%?" "$1" | tail -1 | grep -oE "[0-9.]+"; }
# 抓 RTF（test_wer.py 打印 "平均 RTF: 0.xx" / "Average RTF: 0.xx"；模型加载单独计=热启动）
grab_rtf() { grep -oiE "(平均 |average )?rtf[: ]+[0-9.]+" "$1" | tail -1 | grep -oE "[0-9.]+"; }
record() { echo "asr,$1,$DATASET,$LANG,$2,$3,$4,$5,$6,$7" >> "$CSV"; }

echo ">>> ASR benchmark: dataset=$DATASET lang=$LANG ($(wc -l < "$GT") 条)"

# 可选模型子集（全量分批跑用）：ASR_MODELS="whisper" / "sensevoice firered" ...
WANT="${ASR_MODELS:-whisper sensevoice firered wenet zipformer}"
has_m() { echo " $WANT " | grep -q " $1 "; }

# ---- Whisper ----
if has_m whisper && [ -d "$MODEL_ROOT/Whisper/python" ]; then
    for mt in ${WHISPER_VARIANTS:-tiny base small turbo}; do
        [ -d "$MODEL_ROOT/Whisper/models-$($PY "$REPO_DIR/tools/cfg.py" models.chip 2>/dev/null || echo ax650)" ] || true
        ( cd "$MODEL_ROOT/Whisper/python" && \
          $PY test_wer.py -d aishell --gt_path "$GT" --model_type $mt \
             --model_path ../models-ax650 --language $LANG >$LOGD/w_$mt.log 2>&1 )
        v=$(grab_cer $LOGD/w_$mt.log); r=$(grab_rtf $LOGD/w_$mt.log); record "whisper-$mt" "$([ "$LANG" = zh ] && echo cer || echo wer)" "${v:-NA}" "$r" "" "" "board"
        echo "  whisper-$mt: ${v:-NA}"
    done
fi

# ---- SenseVoice ----
if has_m sensevoice && [ -d "$MODEL_ROOT/SenseVoice/python" ]; then
    ( cd "$MODEL_ROOT/SenseVoice/python" && \
      $PY test_wer.py -d aishell -g "$GT" --language $LANG >$LOGD/sv.log 2>&1 )
    v=$(grab_cer $LOGD/sv.log); r=$(grab_rtf $LOGD/sv.log); record "sensevoice" "$([ "$LANG" = zh ] && echo cer || echo wer)" "${v:-NA}" "$r" "" "" "board"
    echo "  sensevoice: ${v:-NA}"
fi

# ---- FireRedASR-AED ----
if has_m firered && [ -d "$MODEL_ROOT/FireRedASR-AED" ]; then
    ( cd "$MODEL_ROOT/FireRedASR-AED" && \
      $PY test_wer.py -d aishell -g "$GT" --language $LANG >$LOGD/fr.log 2>&1 )
    v=$(grab_cer $LOGD/fr.log); r=$(grab_rtf $LOGD/fr.log); record "firered-aed" cer "${v:-NA}" "$r" "" "" "board"
    echo "  firered-aed: ${v:-NA}"
fi

# ---- WeNet（仅 zh）----
if has_m wenet && [ -d "$MODEL_ROOT/WeNet" ] && [ "$LANG" = zh ]; then
    ( cd "$MODEL_ROOT/WeNet" && \
      $PY test_wer.py -g "$GT" --mode ctc_prefix_beam_search >$LOGD/wenet.log 2>&1 )
    v=$(grab_cer $LOGD/wenet.log); r=$(grab_rtf $LOGD/wenet.log); record "wenet" cer "${v:-NA}" "$r" "" "" "board"
    echo "  wenet: ${v:-NA}"
fi

# ---- Zipformer ----
if has_m zipformer && [ -d "$MODEL_ROOT/Zipformer.axera" ]; then
    ( cd "$MODEL_ROOT/Zipformer.axera" && \
      $PY test_wer.py -g "$GT" >$LOGD/zip.log 2>&1 )
    v=$(grab_cer $LOGD/zip.log); r=$(grab_rtf $LOGD/zip.log); record "zipformer" cer "${v:-NA}" "$r" "" "" "board"
    echo "  zipformer: ${v:-NA}"
fi

echo ">>> ASR 完成 -> $CSV"
