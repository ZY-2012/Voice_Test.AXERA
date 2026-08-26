#!/bin/bash
# 板端后台跑全量基准（VAD+SE+ASR，标准子集）。setsid 脱离 SSH，日志写 results/run_all.log。
# 用法（板端 base 环境）: setsid bash launch_bench.sh &
#   或从 host: sshpass ... "cd ...; setsid bash launch_bench.sh >/dev/null 2>&1 &"
set -u
cd "$(dirname "$0")"
export PICO_SEC="${PICO_SEC:-600}"   # picovoice 长流截取前 600s（约 20h 全量不现实）
LOG=results/run_all.log
mkdir -p results
{
  echo "==== 开始 $(date) PICO_SEC=$PICO_SEC ===="
  bash run_benchmark.sh vad se asr
  echo "==== 结束 $(date) ===="
} > "$LOG" 2>&1
