# TTS 模块（语音合成）

**数据**：LJSpeech（英文，test 500 条）+ AISHELL-3（中文，test 1948 条 / 100 说话人）。
**主指标**：**回环 CER/WER**（合成音经已适配 ASR 识别 vs 输入文本，衡量可懂度，跨模型统一、说话人无关）+ RTF。
MCD 仅在零样本克隆同一参考说话人时有意义，默认不作跨模型比较（`tools/metrics_tts.py` 备用）。

## 已接入模型（AX650N）
| 模型 | 类型 | 语言 | 采样率 | HF | 板端状态 |
|---|---|---|---|---|---|
| Kokoro | 预设音色 | zh/en/ja | 24k | AXERA-TECH/kokoro.axera | ✅ zh+en 跑通（zh 回环CER偏高，见下）|
| MeloTTS | 预设音色 | zh/en/jp | 44.1k | AXERA-TECH/MeloTTS | ✅ zh 回环CER 0% |
| ZipVoice | 零样本(固定prompt) | zh/en | 24k | AXERA-TECH/ZipVoice.AXERA | 已接线待跑 |
| CosyVoice2 | 零样本(C++ LLM) | zh/en | 24k | AXERA-TECH/Cosyvoice2.Axera | 已接线待跑(需 tokenizer server) |

### 板端依赖（TTS，base 环境）
```bash
# melotts
pip install cn2an inflect pykakasi pypinyin g2p_en
# kokoro（英文 g2p 走 misaki.en + espeak；中文走 misaki.zh）
pip install kokoro misaki num2words phonemizer ordered_set mojimoji jaconv
pip install spacy && python -m spacy download en_core_web_sm
apt-get install -y espeak-ng          # phonemizer 后端
```

### kokoro.axera 兼容补丁（misaki 0.7.4 API 漂移，已在板端 inference_utils.py 修）
- `zh.ZHG2P(version=None, en_callable=...)` → misaki0.7.4 的 ZHG2P 无参构造：加 try/except 回退 `zh.ZHG2P()`
- zh g2p `__call__` 新版直接返回 phonemes 字符串（非 2-tuple）：`text_to_phonemes` 兼容 str/tuple 返回
- 注：kokoro 中文合成回环 CER 偏高（~70%，合成音偏短/截断），melotts 中文 0%——属模型质量差异，框架已正确捕获。

## 制作
`prepare_tts.py`（LJSpeech/AISHELL-3 → text + wav.scp，中文剥离拼音）。子集：`benchmark/tts/<ds>/{text, wav.scp}`。

## 评测（板端）
```bash
bash tts/run.sh                                   # config 里 models.tts 全部 × zh/en
TTS_MODELS="kokoro melotts" bash tts/run.sh       # 指定模型
TTS_LIMIT=20 bash tts/run.sh                      # cosyvoice2 等重模型限条数
```
流程（每模型×数据集）：`synth_batch.py` 合成 text→`benchmark/tts/syn/<model>_<ds>/`（记录 RTF）
→ 构建回环评测目录（合成音+输入文本作参考）→ 复用 `SenseVoice/test_wer.py` 得 CER/WER → `results/tts.csv`。

> 零样本模型用各仓库自带固定 prompt（zipvoice: moss_prompts；cosyvoice2: prompt_files）。
> cosyvoice2 为 C++ LLM 二进制、需 tokenizer server，较重，默认限 20 条。指标板端实测后填入。
