# SE 模块测试流程

语音增强（Speech Enhancement）板端 benchmark。适配模型：**GTCRN**、**FastEnhancer**（AX650N/AX630C/AX620Q）。

## 1. 数据集
VoiceBank-DEMAND test：824 对 clean/noisy（16kHz，40 种噪声条件）。标准子集 200 对。

## 2. 制作（host 端）
```bash
bash download_datasets.sh se     # HF: JacobLinCool/VoiceBank-DEMAND-16k (test parquet)
bash prepare_datasets.sh se      # parquet -> clean/ + noisy/ + test.scp; 抽子集 -> benchmark/se/voicebank_demand/subset.scp
```

## 3. 测试（板端 base 环境）
```bash
bash se/run.sh                        # noisy 基线 + config 里 models.se 全部模型
SE_MODELS="gtcrn" bash se/run.sh      # 指定模型
LIMIT=6 SE_MODELS="gtcrn" bash se/run.sh   # 小样冒烟
```
流程：`enhance_warm.py`（**模型载入一次**，进程内循环增强 noisy → `benchmark/se/enhanced/<model>/`）→ `eval_se.py` 算 PESQ/STOI/SI-SNR（vs clean）→ `results/se.csv`。

- 板端依赖：`pip install pesq pystoi`；gtcrn 需 torch(CPU)+librosa（base 已装）。
- gtcrn：复用 demo 的 `init_caches/update_caches` + STFT/帧循环/ISTFT，InferenceSession 载入一次。
- fastenhancer：`FastEnhancer` SDK 载入一次，`enhance()` 逐条。

## 4. 指标口径
- PESQ / STOI / SI-SNR（`tools/metrics_se.py`）；noisy-vs-clean 作增强增益基线。
- **RTF = 推理时间/音频时长，不含模型加载/初始化**（载入一次，跳过首条 warmup，只计增强 pipeline）。

## 5. 板端实测（全量标准子集 200 对，2026-08）
| 模型 | PESQ | STOI | SI-SNR(dB) | RTF(推理only) | 备注 |
|---|---|---|---|---|---|
| noisy 基线 | 1.95 | 0.92 | 8.7 | — | 未增强参照 |
| GTCRN | 2.55 | 0.93 | 13.7 | 0.249 | python（无C++）|
| FastEnhancer | 2.75 | 0.94 | 16.2 | 0.186 | python 0.186；vendor C++ 报 0.02 |

> 两模型均优于基线，FastEnhancer 全指标更佳且更快。
> FastEnhancer 有 C++ 可执行 `bin/fastenhancer_ax650_16k`（vendor 报 RTF 0.02），
> 但其 CLI 不单独输出推理时间（墙钟含加载），故 RTF 以 python 载入一次的推理-only 口径为准。

