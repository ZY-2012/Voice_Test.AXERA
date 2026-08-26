# Voice_Test.AXERA

Axera 平台音频 benchmark：**SE（语音增强）/ VAD / ASR / TTS** 四模块统一测试框架。
目标——客户按 5 步一键复现我们已适配模型的指标。

> 模型与数据**不入库**。本仓库只含脚本+文档；数据下载地址与制作方法见下。

---

## 快速复现（5 步，每步一个脚本）

```bash
bash setup_env.sh                 # ① 装依赖（板端加 --board）
bash download_datasets.sh         # ② 下原始数据（可 asr/vad/se/tts 选下）
bash prepare_datasets.sh          # ③ 制作测试集 + 抽标准子集
bash download_models.sh           # ④ 从 HF AXERA-TECH 拉 axmodel
bash run_benchmark.sh             # ⑤ 板端跑，出 results/summary.md
```

- **数据准备（②③）** 可在任意机器；**模型推理（⑤）** 需 AX 板端（axengine）。
  本地与板端为同一共享存储时数据两侧通用（路径自适应见 `tools/common.py`，数据根目录由 `configs/benchmark.yaml` 配置）。
- 全部路径/参数集中在 `configs/benchmark.yaml`（客户一般只改 `data_root` / `model_root`）。
- 默认用**确定性抽样的标准子集**（边缘板全量太慢）；`FULL=1 bash prepare_datasets.sh` 跑全量。

---

## 目录结构

```
Voice_Test.AXERA/
├── setup_env.sh / download_datasets.sh / prepare_datasets.sh
├── download_models.sh / run_benchmark.sh        # 5 个顶层脚本
├── configs/benchmark.yaml                        # 中心配置
├── tools/                                        # 通用指标库（模块无关）
│   ├── metrics_asr.py  CER/WER          metrics_vad.py  帧级F1/AUC
│   ├── metrics_se.py   PESQ/STOI/SISNR  metrics_tts.py  回环CER/MCD
│   ├── common.py  配置/抽样/RTF/CMM/内存   make_subsets.py  确定性子集
│   └── aggregate_results.py  汇总 results/*.csv → summary.md
├── asr/ vad/ se/ tts/   每模块: prepare*.py + run.sh + eval*.py + README
└── results/             指标输出（csv + summary.md）
```

---

## 四模块数据与指标

| 模块 | 数据集 | 规模(全量) | 标准子集 | 指标 |
|---|---|---|---|---|
| **ASR** | ESB 英文6子集 + AISHELL-1 中文 | 43,858 + 7,176 | en 各100 / zh 200 | CER(zh)/WER(en) |
| **VAD** | Picovoice长流 / LibriVAD / AISHELL-1 | 长流 + 2620 + 7176 | 各 200（长流不抽） | 帧级 F1/acc/AUC |
| **SE**  | VoiceBank-DEMAND | 824 对 | 200 对 | PESQ/STOI/SI-SNR |
| **TTS** | LJSpeech英 + AISHELL-3中 | 500 + 1948 | 各 200 | 回环CER + MCD + RTF |

数据来源、制作细节、缺失说明见各模块 `README.md`；**数据集目录/下载地址/数量速查见仓库根 `数据清单.md`**，完整制作方法见 `tools/数据总结.md`。

---

## 已适配模型指标（AX650N 实测，2026-08）

### ASR（AIShell 子集，字符级 CER）
| 模型 | CER | RTF | CMM/OS(MB) |
|---|---|---|---|
| Whisper Tiny/Base/Small/Turbo | 24 / 18 / 11 / 6 % | 0.08~0.48(C++) | 332~2065 |
| FireRedASR-AED | **0.30%** | ~0.30 | 1244/1504 |
| SenseVoice | 3.22% | 0.014 非流式 | 246/658 |
| WeNet | **1.80%** | 0.12 | 79/94 |
| Zipformer | 12.36% | 0.173 | 41/283 |

### VAD
| 模型 | RTF(流式32ms) | CMM/OS(MB) |
|---|---|---|
| SileroVAD | 0.0297（0.95ms/chunk）| 0.7 / 241.8 |

### SE（VoiceBank-DEMAND 200 对，PESQ/STOI/SI-SNR + RTF）
| 模型 | 采样率 | PESQ | STOI | SI-SNR(dB) | RTF(推理only) |
|---|---|---|---|---|---|
| noisy 基线 | 16k | 1.95 | 0.92 | 8.7 | — |
| GTCRN | 16k | 2.55 | 0.93 | 13.7 | 0.249 |
| FastEnhancer | 16k/48k | **2.75** | **0.94** | **16.2** | **0.186** |

### TTS（回环 CER/WER + RTF；主指标为可懂度，AISHELL-3 中文 200 条级）
| 模型 | 类型 | 语言 | 回环 CER | RTF |
|---|---|---|---|---|
| MeloTTS | 预设音色 | zh/en/jp | **12.4%** | 0.139 |
| ZipVoice | 零样本 | zh/en | 21.7% | NPU 0.155 |
| CosyVoice2 | 零样本(C++ LLM) | zh/en | 29.3% | 3.24（本地盘）|
| Kokoro | 预设音色 | zh/en/ja | 70.6% | 10.1（合成偏短，模型问题）|

> SE/TTS 指标为 AX650N 板端实测（2026-08），测试流程见 `se/README.md`、`tts/README.md`；全量复测 `bash run_benchmark.sh se tts`。

---

## 模型来源

- axmodel 权重：HF `AXERA-TECH/*`（`download_models.sh` 自动拉取，芯片见 config `models.chip`）
- 推理代码：各模型 GitHub 仓库（`ml-inory/*.axera` 等），克隆到 `model_root/<model>/`
- 环境：板端 base conda + axengine；torch 系列装 CPU 版（`setup_env.sh --board`）

## 备注

- gigaspeech 含少量音乐/噪声段（自动对齐特性），ASR 默认可不启用。
- common_voice / spgispeech 因源受限（登录 / gated）未纳入，详见 `tools/数据总结.md`。
- DEMAND 噪声（Picovoice VAD 用）需 Kaggle：`kaggle datasets download -d aanhari/demand-dataset`。
