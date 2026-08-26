# TTS 模块（语音合成）

**数据**：LJSpeech（英文，test 500 条）+ AISHELL-3（中文，test 1948 条 / 100 说话人）。
**主指标**：**回环 CER/WER**（合成音经已适配 ASR 识别 vs 输入文本，衡量可懂度，跨模型统一、说话人无关）+ RTF。
MCD 仅在零样本克隆同一参考说话人时有意义，默认不作跨模型比较（`tools/metrics_tts.py` 备用）。

## 已接入模型（AX650N）
| 模型 | 类型 | 语言 | 采样率 | HF | 板端状态 |
|---|---|---|---|---|---|
| Kokoro | 预设音色 | zh/en/ja | 24k | AXERA-TECH/kokoro.axera | ✅ zh+en 跑通（zh 回环 CER 70.6%，合成偏短/截断）|
| MeloTTS | 预设音色 | zh/en/jp | 44.1k | AXERA-TECH/MeloTTS | ✅ zh 回环 CER 0% |
| ZipVoice | 零样本(固定prompt) | zh/en | 24k | AXERA-TECH/ZipVoice.AXERA | ✅ zh 200 条回环 CER 21.7%，NPU RTF 0.155 |
| CosyVoice2 | 零样本(C++ LLM) | zh/en | 24k | AXERA-TECH/Cosyvoice2.Axera | ✅ zh 回环 CER 25%，需先起 tokenizer server |

## 实测回环 CER（中文 AISHELL-3，回环 ASR=SenseVoice）
| 模型 | CER | 样本数 | RTF | 备注 |
|---|---|---|---|---|
| MeloTTS | **12.4%** | 200 | 0.139（载入一次热启动）| 预设音色最佳 |
| ZipVoice | 21.7% | 200 | NPU 0.155 / 端到端含加载 3.13 | 零样本最佳 |
| CosyVoice2 | 29.3% | 173（161 成功/39 超时）| 本地盘 3.24 | 零样本，需 tokenizer server |
| Kokoro | 70.6% | 3 | 10.1 | 合成偏短/截断（模型质量问题）|

### CosyVoice2 启动步骤（板端）
```bash
# 1) tokenizer server（127.0.0.1:12345，main_ax650 依赖）
cd $MODEL_ROOT/CosyVoice2/scripts
setsid python cosyvoice2_tokenizer.py --host 127.0.0.1 --port 12345 > $REPO/results/logs/tokenizer.log 2>&1 &
# 依赖: pip install --no-deps openai-whisper tiktoken
# 2) 合成（synth_batch 会自动改 output*.wav 到 <id>.wav）
```

### 板端依赖（TTS，base 环境）
```bash
# melotts
pip install cn2an inflect pykakasi pypinyin g2p_en
# kokoro（英文 g2p 走 misaki.en + espeak；中文走 misaki.zh）
pip install kokoro misaki num2words phonemizer ordered_set mojimoji jaconv
pip install spacy && python -m spacy download en_core_web_sm
apt-get install -y espeak-ng          # phonemizer 后端
# zipvoice: pip install jieba
# cosyvoice2: pip install --no-deps openai-whisper tiktoken
```

### 已知问题与兼容补丁
- kokoro.axera × misaki 0.7.4 API 漂移：`inference_utils.py` 已加兼容（ZHG2P 无参构造 + str/tuple 返回），中文合成偏短（模型问题）。
- zipvoice 二进制 "Done!" 后不退出：synth_batch 以 `--timeout` 容忍（输出已生成即算成功）。
- zipvoice 等日志量大：synth_batch 输出重定向到 `syn/<model>_<ds>/_run.log`（避免 PIPE 死锁），已合成文件幂等跳过。

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
