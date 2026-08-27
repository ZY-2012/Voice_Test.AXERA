# TTS 模块（语音合成）

板端文本转语音 benchmark。适配模型：**MeloTTS / Kokoro（预设音色）**、**ZipVoice / CosyVoice2（零样本）**。

## 1. 数据集

| 数据集 | 语言 | 规模 | 采样率 | 下载地址 |
|---|---|---|---|---|
| LJSpeech | 英语 | 500 条（全量 13,100 条 / 24h）| 22.05k | HF `flexthink/ljspeech` |
| AISHELL-3 | 中文多说话人 | 1,948 条 / 100 说话人（官方 test 24,773 条）| 44.1k | HF `AISHELL/AISHELL-3` |

## 2. 制作

`prepare_tts.py`（LJSpeech 按 test.txt 清单、AISHELL-3 中文剥离拼音 → `text` + `wav.scp`）；子集由 `tools/make_subsets.py` 产出 `benchmark/tts/<ds>/{text, wav.scp}`。

## 3. 测试命令（板端 base 环境）

```bash
bash tts/run.sh                                # config 里 models.tts 全部 × zh/en
TTS_MODELS="kokoro melotts" bash tts/run.sh    # 指定模型
TTS_LIMIT=20 bash tts/run.sh                   # 限条数（重模型用）
```

流程：`synth_batch.py` 合成 text → `benchmark/tts/syn/<model>_<ds>/`（记录 RTF）→ 构建回环目录（合成音+输入文本作参考）→ 复用 `SenseVoice/test_wer.py` 得回环 CER/WER → `results/tts.csv`。

**CosyVoice2 前置**（C++ LLM，需 tokenizer server 127.0.0.1:12345）：

```bash
cd CosyVoice2/scripts && nohup python cosyvoice2_tokenizer.py --host 127.0.0.1 --port 12345 &
# 板端建议拷贝模型到本地盘跑（NFS 链路慢）：COSYVOICE2_DIR=/root/cosyvoice2_local/CosyVoice2
```

**板端依赖**：melotts（cn2an/inflect/pykakasi/pypinyin/g2p_en）；kokoro（kokoro/misaki/num2words/phonemizer/ordered_set/mojimoji/jaconv + `pip install spacy && python -m spacy download en_core_web_sm` + `apt install espeak-ng`）；zipvoice（jieba）；cosyvoice2（`pip install --no-deps openai-whisper tiktoken`）。

## 4. 指标口径

- **回环 CER/WER**：合成音经已适配 ASR（默认 sensevoice）识别 vs 输入文本（可懂度，越低越好，跨模型统一）
- RTF = 推理时间/音频时长，**不含模型加载**（MeloTTS 用载入一次驱动 `melotts_batch.py`）
- MCD（梅尔倒谱距离）仅在零样本克隆同一参考说话人时可比，默认不作跨模型比较（`tools/metrics_tts.py` 备用）

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- 零样本模型用各仓库自带固定 prompt（zipvoice: moss_prompts；cosyvoice2: prompt_files）
- 已知问题：kokoro 中文合成偏短/截断（回环 CER 高，模型质量问题）；zipvoice 二进制完成后不退出（synth_batch 已加超时容忍）；cosyvoice2 长文本慢路径会超时（实测 161/200 成功）
- 板端依赖详见 [audio-benchmark skill](.claude/skills/audio-benchmark/SKILL.md) 已知坑
