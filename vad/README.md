# VAD 模块

语音活动检测板端 benchmark。适配模型：**SileroVAD**（python 包，AX650N/AX630C）、**TEN VAD**（C++ 可执行，AX650）。

## 1. 数据集（4 套；英文语音基于 LibriSpeech test-clean，中文基于 AISHELL-1）

| 数据集 | 标签 | 规模 | 语言 |
|---|---|---|---|
| picovoice | 32ms 帧三值(0静音/1未知忽略/2语音) | 2620 句拼接长流 + DEMAND 0dB 混噪 | en |
| librivad | 采样级二值(0/1)，强制对齐 TextGrid 导出 | 2620 条 | en |
| aishell1 | 采样级二值(0/1)，funasr fsmn-vad 生成 | 7176 条 | zh |
| tenvad | 采样级二值(0/1)，TEN VAD 官方 scv 段标注转换 | 30 条 / 4.4min | en |

## 2. 制作

`gen_picovoice_test.py`（官方 mixer.py 逻辑）、`gen_librivad_test.py`（官方 create_labels.py 逻辑）、
`gen_aishell_vad_labels.py`（funasr）、`gen_tenvad_test.py`（TEN VAD 模型仓库自带 testset 的 scv→npy）。
子集：`tools/make_subsets.py` → `benchmark/vad/*/subset.scp`（picovoice 长流不抽，tenvad 仅 30 条即全量）。

## 3. 测试命令（板端 base 环境）

```bash
bash vad/run.sh                                   # 全部模型（清单来自 config models.vad）
VAD_MODELS=tenvad bash vad/run.sh                 # 只跑指定模型

# SileroVAD（依赖: pip install --no-deps -e $MODEL_ROOT/silero_vad_axera-0.1.2）
python vad/eval_silero.py --backend ax650         # 四数据集
LIMIT=5 python vad/eval_silero.py --backend ax650 # 小样冒烟
PICO_SEC=600 python vad/eval_silero.py ...        # picovoice 长流截取前 N 秒

# TEN VAD（先在 host 端交叉编译一次；需 AX650 BSP SDK + gcc-aarch64-linux-gnu）
bash vad/build_tenvad.sh [/path/to/ax650n_bsp_sdk/msp/out]
python vad/eval_tenvad.py [--dataset tenvad librivad aishell1 picovoice]
```

SileroVAD 用**流式 chunk API**（`model(chunk,sr)` 每 512 采样=32ms 出一帧概率），贴近真实流式 VAD，
且避开 vendor `audio_forward` 的 np.pad bug。TEN VAD 无 python 绑定，逐条调 `ten_vad_example`
（256 采样=16ms 帧移）并解析 stdout 的每帧概率与自报 RTF。两者结果都写 `results/vad.csv`。

## 4. 指标口径

- 帧级 accuracy/precision/recall/F1 + ROC-AUC（`tools/metrics_vad.py`）；picovoice 的"未知"帧(标签1)不计
- **RTF = 推理时间/音频时长，不含模型加载**（silero 载入一次只计 `_frame_probs`；TEN VAD 取
  可执行内部计时，其起点在 `ten_vad_create` 之后，与本工程口径一致）
- 帧长随模型：silero 32ms、TEN VAD 16ms（帧级 P/R/F1 对帧长不敏感，CSV note 列记录实际值）；
  picovoice 参考标签固定 32ms 栅格，TEN VAD 的 16ms 概率按 `prob_to_frames` 重采样对齐

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- aishell1 F1 相对较低因其参考标签由 fsmn-vad 模型生成（模型间差异），silero precision=1.0 说明它更保守
- SileroVAD RTF≈0.029 与 vendor 报告 0.0297 吻合
- TEN VAD 源码把模型路径写死为相对路径 `axmodel/ten-vad-ax650.axmodel`，故 `eval_tenvad.py`
  以代码仓库目录为 cwd 运行；`build_tenvad.sh` 会把模型仓库里的 axmodel 软链到该位置
- TEN VAD 逐条起子进程（每次含模型加载），200 条约 17 分钟；RTF 取可执行自报值，不含子进程开销
- 可执行必须写一路立体声输出（L=VAD flag），逐样本 fwrite。这份临时 wav 默认落**本机盘**
  `~/.tenvad_scratch`（`TENVAD_SCRATCH` 可覆盖）——放共享 NFS 上会让 I/O 成为瓶颈（板端链路仅 ~2.7MB/s）
- OS 内存取子进程 `/proc/<pid>/status` 的 `VmHWM`（内核维护的峰值）。不能用
  `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`：它是历来所有子进程的最大值，单调不减且混入
  python 自身 fork 开销（实测 11.8MB vs 真实 4.0MB）
- picovoice 那行的内存明显偏高，是因为可执行会把整段截取音频读进内存（600s ≈ 19MB），与模型无关
- DEMAND 噪声需 Kaggle（`aanhari/demand-dataset`）
