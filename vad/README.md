# VAD 模块测试流程

语音活动检测（Voice Activity Detection）板端 benchmark。适配模型：**SileroVAD**（AX650N/AX630C）。

## 1. 数据集（3 套，语音均基于 LibriSpeech test-clean / AISHELL-1）

| 数据集 | 标签 | 规模 | 语言 |
|---|---|---|---|
| picovoice | 32ms 帧三值(0静音/1未知忽略/2语音) | 2620 句拼接长流 + DEMAND 0dB 混噪 | en |
| librivad | 采样级二值(0/1)，强制对齐 TextGrid 导出 | 2620 条 | en |
| aishell1 | 采样级二值(0/1)，funasr fsmn-vad 生成 | 7176 条 | zh |

## 2. 制作（host 端）
```bash
bash download_datasets.sh vad        # DEMAND(Kaggle)、LibriVAD 对齐/噪声
bash prepare_datasets.sh vad         # 生成 picovoice 长流 + librivad/aishell 采样级标签
                                     # 抽标准子集 -> benchmark/vad/{librivad,aishell1}/subset.scp + picovoice/info.txt
```

## 3. 测试（板端 base 环境）
```bash
# 依赖: pip install --no-deps -e $MODEL_ROOT/silero_vad_axera-0.1.2
python vad/eval_silero.py --backend ax650                 # 三数据集
python vad/eval_silero.py --backend ax650 --dataset librivad   # 单数据集
LIMIT=5 python vad/eval_silero.py --backend ax650         # 小样冒烟
PICO_SEC=600 ...                                          # picovoice 长流截取前 N 秒(默认600)
```
- 推理用**流式 chunk API**（`model(chunk,sr)` 每 512 采样=32ms 出一帧概率），贴近真实流式 VAD，且避开 vendor `audio_forward` 的 np.pad bug。
- 结果写 `results/vad.csv`（f1/accuracy/precision/recall/roc_auc + rtf + os_mb）。

## 4. 指标口径
- 帧级 accuracy/precision/recall/F1 + ROC-AUC（`tools/metrics_vad.py`）；picovoice 的"未知"帧(标签1)不计。
- **RTF = 推理时间/音频时长，不含模型加载**（silero 载入一次，只计 `_frame_probs`）。

## 5. 板端实测（全量标准子集 200×2 + picovoice 600s，2026-08）
| 数据集 | F1 | AUC | acc | RTF(推理only) |
|---|---|---|---|---|
| librivad(en) | 0.966 | 0.977 | 0.944 | 0.0295 |
| aishell1(zh) | 0.878 | 0.905 | 0.794 | 0.0312 |
| picovoice(en) | 0.959 | 0.997 | 0.986 | 0.0286 |

> aishell1 F1 较低因参考标签由 fsmn-vad 模型生成（模型间差异），precision=1.0 说明 silero 更保守。
> SileroVAD 无 C++ 可执行；RTF≈0.029 与 vendor 报告 0.0297 吻合。OS内存≈45MB。

