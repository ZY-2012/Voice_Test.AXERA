# SE 模块（语音增强）

语音增强板端 benchmark。适配模型：**GTCRN**、**FastEnhancer**、**DeepFilterNet3**、**GCRN**（AX650N/AX630C/AX620Q）。

## 1. 数据集

VoiceBank-DEMAND test：824 对 clean/noisy（16kHz，40 种噪声条件）。标准子集 200 对。

## 2. 制作

`prepare_se.py`（parquet → clean/ + noisy/ + test.scp）。子集：`benchmark/se/voicebank_demand/subset.scp`。

## 3. 测试命令（板端 base 环境）

```bash
bash se/run.sh                        # noisy 基线 + config 里 models.se 全部模型
SE_MODELS="gtcrn" bash se/run.sh      # 指定模型
LIMIT=6 SE_MODELS="gtcrn" bash se/run.sh   # 小样冒烟
```

流程：`enhance_warm.py`（**模型载入一次**，进程内循环增强 noisy → `benchmark/se/enhanced/<model>/`）
→ `eval_se.py` 算 PESQ/STOI/SI-SNR（vs clean）→ `results/se.csv`。
板端依赖：`pip install pesq pystoi`；gtcrn 需 torch(CPU)+librosa。

## 4. 指标口径

- PESQ / STOI / SI-SNR（`tools/metrics_se.py`）；noisy-vs-clean 作增强增益基线
- **RTF = 推理时间/音频时长，不含模型加载**（载入一次，跳过首条 warmup，只计增强 pipeline）

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- FastEnhancer 有 C++ 可执行 `bin/fastenhancer_ax650_16k`（vendor 报 RTF 0.02），
  但其 CLI 不单独输出推理时间（墙钟含加载），故 RTF 以 python 载入一次的推理-only 口径为准
- GTCRN 无 C++ 可执行
- DeepFilterNet3 为 **48kHz 模型**：16k noisy 重采样 48k 增强后降回 16k 评估（`enhance_warm.py` 内置）；RTF 为三模型最快，STOI 略降为 16k↔48k 重采样特性
- GCRN SDK 的 `enhance()` 要求 **int16 PCM 输入**（float 输入会导致输出崩坏，已踩坑）
