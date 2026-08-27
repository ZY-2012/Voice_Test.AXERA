# ASR 模块

语音识别板端 benchmark。适配模型：**Whisper**(tiny/base/small/turbo)、**SenseVoice**、**FireRedASR-AED**、**WeNet**、**Zipformer**。

## 1. 数据集

| 数据集 | 语言 | 规模 | 标准子集 |
|---|---|---|---|
| AISHELL-1 | 中文 | 7,176 | 200 |
| ESB 6 子集(librispeech/voxpopuli/tedlium/earnings22/ami/gigaspeech) | 英文 | 43,858 | 各100 |

## 2. 制作

`prepare_asr_test.py`（ESB tar/parquet→16k wav+wav.scp+text）、`prepare_aishell.py`（AISHELL-1 整理）。
子集由 `tools/make_subsets.py` 产出 `benchmark/asr/<ds>/{ground_truth.txt, wav/}`，并同时提供
`aishell_S0764/`(软链) 两种音频目录名——FireRedASR 用 `wav/`，Whisper/SenseVoice/WeNet 用 `aishell_S0764/`。

## 3. 测试命令（板端 base 环境）

```bash
bash asr/run.sh                          # 默认中文 aishell1 子集，跑全部模型
DATASET=librispeech bash asr/run.sh      # 换英文子集
ASR_MODELS="sensevoice wenet" bash asr/run.sh   # 指定模型
LIMIT=3 bash asr/run.sh                  # 小样冒烟
```

逐模型调用其仓库 `test_wer.py` 指向 benchmark 子集，抓 CER/WER + RTF 写 `results/asr.csv`。板端依赖：`pip install zhconv`（Whisper）。

## 4. 指标口径

- 中文 CER（字符级）/ 英文 WER（词级），去标点转小写（各仓库 test_wer.py 内置，与 `tools/metrics_asr.py` 一致）
- **RTF = 推理时间/音频时长，不含模型加载**：C++ 用 CLI 自报（whisper_cli/test_sensevoice），python 用载入一次探针 `asr/rtf_probe.py`

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- Whisper 个别音频触发 test_wer.py 的 UTF-8 解码异常，已加 try/except 跳过（计为该条错误）
- whisper_cli 对个别音频重复解码（vendor bug）；C++ 与 python RTF 差异：tiny 因 CPU 特征提取占主导两者相当，base/small/turbo C++ 更低
- WeNet 需 `pretrained/units.txt` 为 AIShell-1 版（4233 词，板端已确认正确）
- FireRedASR 无 C++ 预编译二进制（仅 build.sh），RTF 用 python 值
