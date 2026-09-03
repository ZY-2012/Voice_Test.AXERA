# Voice_Test.AXERA

Axera 平台音频模型 benchmark：**ASR / VAD / SE / TTS** 四模块统一测试框架。
目标——使用者按 5 步一键复现已适配模型的板端指标。

> 模型与数据**不入库**。本仓库只含脚本+文档；数据目录/数量/下载地址速查见 [数据清单.md](数据清单.md)，制作方法与已知限制见 `tools/数据总结.md`。

## 目录

- [快速复现（5 步）](#快速复现5-步每步一个脚本)
- [目录结构](#目录结构)
- [模块总览](#模块总览)
- [模块就绪度](#模块就绪度)
- [已适配模型指标](#已适配模型指标自动生成勿手改)
- [模型来源](#模型来源)
- [备注](#备注)

---

## 快速复现（5 步，每步一个脚本）

```bash
bash setup_env.sh                 # ① 装依赖（板端加 --board）        → 环境就绪
bash download_datasets.sh         # ② 下原始数据（asr/vad/se/tts 选下）→ data_root/<module>/
bash prepare_datasets.sh          # ③ 制作测试集 + 抽标准子集          → data_root/benchmark/
bash download_models.sh           # ④ 从 HF AXERA-TECH 拉 axmodel     → model_root/<model>/
bash run_benchmark.sh             # ⑤ 板端跑分                        → results/*.csv + summary.md
```

- **数据准备（②③）** 可在任意机器；**模型推理（⑤）** 需 AX 板端（axengine）。
  本地与板端为同一共享存储时数据两侧通用（路径自适应见 `tools/common.py`，数据根目录由 `configs/benchmark.yaml` 配置）。
- 全部路径/参数集中在 `configs/benchmark.yaml`（使用者一般只改 `paths.data_root` / `paths.model_root`）。
- 默认用**确定性抽样的标准子集**（边缘板全量太慢）；`FULL=1 bash prepare_datasets.sh` 跑全量。
- 板端全量长任务可用 `bash launch_bench.sh`（setsid 脱离 SSH 后台跑，日志 `results/run_all.log`）。
- 跑分完成后在仓库根执行 `python tools/report.py --readme`，把指标刷新进本 README（提交前必做）。

---

## 目录结构

```
Voice_Test.AXERA/
├── setup_env.sh / download_datasets.sh / prepare_datasets.sh
├── download_models.sh / run_benchmark.sh / launch_bench.sh   # 6 个顶层脚本（①~⑤ + 板端后台启动器）
├── configs/benchmark.yaml                        # 中心配置（路径/子集/模型/模块/口径）
├── 数据清单.md / CONTRIBUTING.md                  # 数据速查 / 扩展指南
├── tools/                                        # 通用库（模块无关）
│   ├── metrics_asr.py  CER/WER          metrics_vad.py  帧级F1/AUC
│   ├── metrics_se.py   PESQ/STOI/SISNR  metrics_tts.py  回环CER/MCD
│   ├── common.py  配置/抽样/RTF/CMM/内存   make_subsets.py  确定性子集
│   ├── model_registry.py  SE/TTS 模型注册表（新增模型改这里）
│   └── report.py  汇总渲染 + README 注入（指标唯一事实来源）
├── asr/ vad/ se/ tts/   每模块: prepare*.py + run.sh + eval*.py + README
├── results/             指标输出（csv + summary.md，不入库）
└── .claude/skills/      内部运维 skill
```

---

## 模块总览

| 模块 | 数据集 | 规模(全量) | 标准子集 | 指标 |
|---|---|---|---|---|
| **ASR** | ESB 英文6子集 + AISHELL-1 中文 | 43,858 + 7,176 | en 各100 / zh 200 | CER(zh)/WER(en) |
| **VAD** | Picovoice长流 / LibriVAD / AISHELL-1 / TEN VAD官方 | 长流 + 2620 + 7176 + 30 | 逐句各 200 + TEN 30（长流不抽）| 帧级 F1/acc/AUC |
| **SE**  | VoiceBank-DEMAND | 824 对 | 200 对 | PESQ/STOI/SI-SNR |
| **TTS** | LJSpeech英 + AISHELL-3中 | 500 + 1948 | 各 200 | 回环CER + MCD + RTF |

每模块测试流程见各自 `README.md`；数据速查见 [数据清单.md](数据清单.md)。

---

## 模块就绪度

<!-- STATUS-MATRIX -->
| 模块 | 数据就绪 | 模型注册 | 指标已测 |
|---|---|---|---|
| 语音识别 ASR | ✅ | ✅ | ✅ |
| 语音活动检测 VAD | ✅ | ✅ | ✅ |
| 语音增强 SE | ✅ | ✅ | ✅ |
| 语音合成 TTS | ✅ | ✅ | ✅ |
<!-- /STATUS-MATRIX -->

---

## 已适配模型指标（自动生成，勿手改）

<!-- RESULTS:asr -->
**语音识别 ASR**
> 口径：AISHELL-1 中文 200 条 CER / ESB 英文各 100 条 WER · RTF 不含加载 · ax650

| 模型 | CER | RTF |
|---|---|---|
| firered-aed | **0.57%** | 0.275 |
| sensevoice | 5.51% | **0.039** |
| wenet | 5.58% | 0.132 |
| whisper-turbo | 10.07% | 0.462 |
| zipformer | 13.22% | 0.205 |
| whisper-small | 15.04% | 0.256 |
| whisper-base | 22.35% | 0.109 |
| whisper-tiny | 42.86% | 0.083 |

> 最优：firered-aed CER=0.57%
<!-- /RESULTS:asr -->

<!-- RESULTS:vad -->
**语音活动检测 VAD**
> 口径：逐句集各 200 条 + TEN VAD 官方 30 条 + Picovoice 长流 · 帧级指标 · ax650

| 模型 | 数据集 | F1 | 准确率 | 精确率 | 召回率 | AUC | RTF | OS(MB) |
|---|---|---|---|---|---|---|---|---|
| silero | librivad | **0.966** | 0.944 | 0.969 | 0.964 | 0.977 | 0.029 | 45 |
| tenvad | librivad | 0.962 | 0.935 | 0.956 | 0.968 | 0.975 | 0.098 | 4 |
| silero | picovoice | 0.959 | **0.986** | 0.990 | 0.930 | **0.997** | **0.029** | 45 |
| silero | tenvad | 0.937 | 0.905 | 0.938 | 0.937 | 0.956 | 0.029 | 47 |
| tenvad | tenvad | 0.931 | 0.898 | 0.941 | 0.922 | 0.949 | 0.097 | 4 |
| tenvad | picovoice | 0.912 | 0.966 | 0.850 | **0.983** | 0.996 | 0.097 | 22 |
| tenvad | aishell1 | 0.891 | 0.814 | 0.998 | 0.805 | 0.908 | 0.098 | **4** |
| silero | aishell1 | 0.878 | 0.794 | **1.000** | 0.783 | 0.905 | 0.031 | 45 |

> 最优：silero F1=0.966
<!-- /RESULTS:vad -->

<!-- RESULTS:se -->
**语音增强 SE**
> 口径：VoiceBank-DEMAND 200 对 · PESQ/STOI/SI-SNR · RTF 不含加载 · ax650

| 模型 | PESQ | STOI | SI-SNR | RTF |
|---|---|---|---|---|
| fastenhancer | **2.75** | **0.936** | 16.2 dB | 0.186 |
| gtcrn | 2.55 | 0.925 | 13.7 dB | 0.249 |
| gcrn | 2.20 | 0.923 | **16.8 dB** | **0.051** |
| deepfilternet3 | 2.13 | 0.881 | 13.3 dB | 0.101 |
| noisy-baseline | 1.95 | 0.924 | 8.7 dB | — |

> 最优：fastenhancer PESQ=2.75
<!-- /RESULTS:se -->

<!-- RESULTS:tts -->
**语音合成 TTS**
> 口径：AISHELL-3 中文 200 条回环 CER · RTF 不含加载 · ax650

| 模型 | 数据集 | 回环CER | GT锚点CER | 替换率(GT) | 删除率(GT) | 插入率(GT) | 替换率 | 删除率 | 插入率 | 成功率 | 均长 | 时长比 | RTF(含加载) | 回环WER | GT锚点WER | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zipvoice | zh_long | **0.58%** | — | — | — | — | **0.58%** | **0.00%** | **0.00%** | **100.0%** | 10.98s | — | **0.59** | — | — | 0.196 |
| melotts | zh_long | 1.64% | — | — | — | — | 1.59% | 0.05% | 0.00% | 100.0% | 12.46s | — | — | — | — | **0.134** |
| cosyvoice2 | zh_long | 1.84% | — | — | — | — | 1.10% | 0.28% | 0.46% | 100.0% | 13.53s | — | 1.20 | — | — | — |
| zipvoice | aishell3 | 4.98% | — | — | — | — | 4.90% | 0.08% | 0.00% | 100.0% | 1.30s | 0.588 | 3.18 | — | — | 0.155 |
| melotts | aishell3 | 8.22% | — | — | — | — | 7.63% | 0.50% | 0.08% | 100.0% | 1.93s | 0.968 | — | — | — | 0.140 |
| cosyvoice2 | aishell3 | 13.43% | — | — | — | — | 5.22% | 8.21% | 0.00% | 90.0% | 1.92s | 0.825 | 556.59 | — | — | 3.242 |
| cosyvoice2-ts4 | aishell3 | 20.90% | — | — | — | — | 6.72% | 14.18% | 0.00% | 75.0% | 2.29s | 0.980 | — | — | — | — |
| kokoro | aishell3 | 70.59% | — | — | — | — | — | — | — | — | — | — | — | — | — | 10.094 |
| GT | aishell3 | — | 2.90% | 2.66% | **0.25%** | **0.00%** | — | — | — | — | — | — | — | — | — | — |
| cosyvoice2 | librispeech | — | — | — | — | — | 26.84% | 5.41% | 3.25% | 100.0% | 11.47s | 1.342 | 1.31 | 35.50% | — | — |
| cosyvoice2 | ljspeech | — | — | — | — | — | 5.69% | 0.60% | 1.80% | 100.0% | 7.61s | 1.268 | 1.45 | 8.08% | — | — |
| GT | librispeech | — | — | **1.48%** | 0.33% | 0.23% | — | — | — | — | — | — | — | — | 2.04% | — |
| melotts | librispeech | — | — | — | — | — | 8.87% | 0.75% | 1.80% | 100.0% | 8.85s | 1.178 | — | 11.43% | — | 0.196 |
| zipvoice | librispeech | — | — | — | — | — | 8.50% | 8.52% | 0.91% | 100.0% | 5.07s | 0.779 | 1.63 | 17.93% | — | 0.221 |
| melotts | ljspeech | — | — | — | — | — | 11.50% | 0.89% | 3.49% | 100.0% | 7.79s | 1.223 | — | 15.88% | — | 0.158 |
| zipvoice | ljspeech | — | — | — | — | — | 2.92% | 0.24% | 2.71% | 100.0% | 5.83s | 0.910 | 1.41 | **5.87%** | — | 0.204 |

> 最优：zipvoice 回环CER=0.58%
<!-- /RESULTS:tts -->

> 以上表格由 `tools/report.py` 从 `results/*.csv` 自动生成。修改指标请改 CSV 后重跑 `python tools/report.py --readme`。

---

## 模型来源

- axmodel 权重：**HF [AXERA-TECH](https://huggingface.co/AXERA-TECH)**（本工程全部模型均在该组织下，`download_models.sh` 自动拉取，芯片见 config `models.chip`）
- 推理代码：各模型 GitHub 仓库（`ml-inory/*.axera` 等），**需手动 clone 到 `model_root/<model>/`**（download_models.sh 只下 axmodel 权重）
- 环境：板端 base conda + axengine；torch 系列装 CPU 版（`setup_env.sh --board`）

## 备注

- gigaspeech 含少量音乐/噪声段（自动对齐特性），ASR 默认可不启用。
- common_voice / spgispeech 因源受限（登录 / gated）未纳入，详见 `tools/数据总结.md`。
- DEMAND 噪声（Picovoice VAD 用）需 Kaggle：`kaggle datasets download -d aanhari/demand-dataset`。
- 新增数据集/模型/模块的步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。
