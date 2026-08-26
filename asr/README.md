# ASR 模块测试流程

语音识别板端 benchmark。适配模型：**Whisper**(tiny/base/small/turbo)、**SenseVoice**、**FireRedASR-AED**、**WeNet**、**Zipformer**。

## 1. 数据集
| 数据集 | 语言 | 规模 | 标准子集 |
|---|---|---|---|
| AISHELL-1 | 中文 | 7176 | 200 |
| ESB 6 子集(librispeech/voxpopuli/tedlium/earnings22/ami/gigaspeech) | 英文 | 43858 | 各100 |

## 2. 制作（host 端）
```bash
bash download_datasets.sh asr      # ESB(aria2c) + AISHELL-1(modelscope)
bash prepare_datasets.sh asr       # 解压→16k wav+text; 抽子集 -> benchmark/asr/<ds>/{ground_truth.txt, wav/, aishell_S0764->wav}
```
> 子集同时提供 `wav/` 与 `aishell_S0764/`(软链) 两种音频目录名，兼容各模型 test_wer.py 约定
> （FireRedASR 用 wav/；Whisper/SenseVoice/WeNet 用 aishell_S0764/）。

## 3. 测试（板端 base 环境）
```bash
bash asr/run.sh                          # 默认中文 aishell1 子集，跑全部 5 模型
DATASET=librispeech bash asr/run.sh      # 换英文子集
LIMIT=3 bash asr/run.sh                  # 小样冒烟
```
逐模型调用其仓库 `test_wer.py` 指向 benchmark 子集，抓 CER/WER + RTF 写 `results/asr.csv`。
- 板端依赖：`pip install zhconv`（Whisper 需要）；FireRedASR 需 kaldiio + 内嵌 silero。
- WeNet 需 `pretrained/units.txt` 为 AIShell-1 版(4233词)——板端已确认正确（输出正常中文，非乱码）。

## 4. 指标口径
- 中文 CER（字符级）/ 英文 WER（词级），去标点转小写（各仓库 test_wer.py 内置，与 `tools/metrics_asr.py` 一致）。
- **RTF = 推理时间/音频时长，不含模型加载**（test_wer.py 将模型加载单独计时，RTF 为推理-only）。

## 5. 板端实测（全量标准子集 200 条，AISHELL 中文 CER，2026-08）
| 模型 | CER | RTF | 说明 |
|---|---|---|---|
| FireRedASR-AED | **0.57%** | 0.275 (py) | 最优；无 C++ 二进制（仅 build.sh）|
| SenseVoice | **5.51%** | C++ 0.039 / py 0.043 | 两者接近 |
| WeNet | **5.58%** | 0.132 (py) | 输出正常中文 |
| Whisper turbo/small/base/tiny | 10.07 / 15.04 / 22.35 / 42.86% | C++ 0.46/0.26/0.11/0.08 vs py 0.51/0.33/0.12/0.07 | 规模越大越好；C++ 普遍更低，tiny 因 CPU 特征提取占主导两者相当 |
| Zipformer | 13.22% | 0.205 (py) | — |

> RTF 口径：推理时间/音频时长，不含模型加载。C++ 值来自 `whisper_cli`/`test_sensevoice` 自报
> （同 9 条正常音频）；python 值来自 `asr/rtf_probe.py`（单进程载入一次，warmup 1 条后计时 9 条）。
> whisper_cli 对个别音频会触发重复解码（vendor bug，第 1 条 AISHELL 音频可复现），
> CER 用 python test_wer（已加 try/except 跳过坏样本），RTF 在正常音频上测。

