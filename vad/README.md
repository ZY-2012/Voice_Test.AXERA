# VAD 模块

语音活动检测板端 benchmark。适配模型：**SileroVAD**（AX650N/AX630C）。

## 1. 数据集（3 套，语音均基于 LibriSpeech test-clean / AISHELL-1）

| 数据集 | 标签 | 规模 | 语言 |
|---|---|---|---|
| picovoice | 32ms 帧三值(0静音/1未知忽略/2语音) | 2620 句拼接长流 + DEMAND 0dB 混噪 | en |
| librivad | 采样级二值(0/1)，强制对齐 TextGrid 导出 | 2620 条 | en |
| aishell1 | 采样级二值(0/1)，funasr fsmn-vad 生成 | 7176 条 | zh |

## 2. 制作

`gen_picovoice_test.py`（官方 mixer.py 逻辑）、`gen_librivad_test.py`（官方 create_labels.py 逻辑）、
`gen_aishell_vad_labels.py`（funasr）。子集：`tools/make_subsets.py` → `benchmark/vad/*/subset.scp`（picovoice 长流不抽）。

## 3. 测试命令（板端 base 环境）

```bash
# 依赖: pip install --no-deps -e $MODEL_ROOT/silero_vad_axera-0.1.2
python vad/eval_silero.py --backend ax650                 # 三数据集
LIMIT=5 python vad/eval_silero.py --backend ax650         # 小样冒烟
PICO_SEC=600 python vad/eval_silero.py ...                # picovoice 长流截取前 N 秒
bash vad/run.sh                                           # 评测+RTF/内存基准
```

推理用**流式 chunk API**（`model(chunk,sr)` 每 512 采样=32ms 出一帧概率），贴近真实流式 VAD，
且避开 vendor `audio_forward` 的 np.pad bug。结果写 `results/vad.csv`。

## 4. 指标口径

- 帧级 accuracy/precision/recall/F1 + ROC-AUC（`tools/metrics_vad.py`）；picovoice 的"未知"帧(标签1)不计
- **RTF = 推理时间/音频时长，不含模型加载**（silero 载入一次，只计 `_frame_probs`）

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- aishell1 F1 相对较低因其参考标签由 fsmn-vad 模型生成（模型间差异），precision=1.0 说明 silero 更保守
- SileroVAD 无 C++ 可执行；RTF≈0.029 与 vendor 报告 0.0297 吻合
- DEMAND 噪声需 Kaggle（`aanhari/demand-dataset`）
