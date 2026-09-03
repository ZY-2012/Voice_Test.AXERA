# TTS 模块（语音合成）

板端文本转语音 benchmark。适配模型：**MeloTTS / Kokoro（预设音色）**、**ZipVoice / CosyVoice2（零样本）**。

## 1. 数据集

**中英分测**：英文集报 WER，中文集报 CER，两侧指标不混算（口径见 §4）。

| 数据集 | 语言 | 规模 | 采样率 | 下载地址 | 状态 |
|---|---|---|---|---|---|
| LJSpeech | 英语（单说话人）| 500 条（全量 13,100 条 / 24h）| 22.05k | HF `flexthink/ljspeech` | ✅ 就绪 |
| AISHELL-3 | 中文多说话人 | 1,948 条 / 100 说话人（官方 test 24,773 条）| 44.1k | HF `AISHELL/AISHELL-3` | ✅ 就绪 |
| **LibriSpeech test-clean** | 英语**多说话人** | **2,620 条 / 40 说话人** | 16k | **复用 `asr/librispeech/`**（零下载）| ✅ 就绪 |
| **zh_hardcase** | 中文难例 | **150 句**（绕口令 40 / 长句 40 / 数字 35 / 同音叠字 35）| — 纯文本 | **仓库内 `tts/zh_hardcase.tsv`** | ✅ 就绪 |
| **zh_long** | 中文长句 | **40 句**（= 难例集的 `hard_long_*`，均 51.7 字）| — 纯文本 | 同上，`prepare_tts.py` 拆出 | ✅ 就绪 |

- **LibriSpeech test-clean**：英文多说话人集（LJSpeech 只有 1 位说话人）。`asr/librispeech/` 混放了 test-clean(2620) + test-other(2939)，原始 tar 与 SPEAKERS.TXT 已不在盘，故用仓库内 `tts/librispeech_testclean_spk.txt`（40 个说话人 ID）过滤 —— 该清单由 `vad/librivad`（仅覆盖 test-clean）反查得出并交叉验证（补集恰为 2939 = test-other 官方条数）。
  > ⚠️ LibriSpeech 转写是**全大写无标点**（`LOOK YONDER DO YOU NOT SEE...`），可能影响部分 TTS 前端的韵律与断句；需要规范文本时用 LJSpeech（有大小写与标点）。CER/WER 计算双侧统一小写去标点，不影响可比性。
- **zh_hardcase** 是版本化的测试定义（随仓库走，非下载数据），`prepare_tts.py` 物化到 `data_root/tts/zh_hardcase/`。产出 `text`（TTS 输入，含标点助韵律）+ `text.ref`（CER 参考）。**数字类的两者不同**：输入是书面形式 `3.8%`，参考是口语形式 `百分之三点八` —— 因 ASR 转写为汉字，用书面形式比对会全错。无参考音频，故不适用 MCD/GT 锚点。
- **zh_long** 是 **RTF 主口径的测量集**（`tools/common.py` 的 `RTF_REFERENCE_DATASET`）。原因见 §4.7：固定 shape 的 axmodel 每条推理耗时恒定，短句集会让 RTF 虚高数倍，只有长句集才与各仓库官方值同量级可比。

## 2. 制作

`prepare_tts.py`（LJSpeech 按 test.txt 清单、AISHELL-3 剥离拼音、LibriSpeech 按说话人过滤 test-clean、zh_hardcase/zh_long 从仓库 tsv 物化 → `text` [+ `text.ref`] + `wav.scp`）；子集由 `tools/make_subsets.py` 产出 `benchmark/tts/<ds>/`（zh_hardcase 150 / zh_long 40，均无 `wav.scp`）。

## 3. 测试命令（板端 base 环境）

```bash
bash tts/run.sh                                      # config 里 enabled 的 models.tts 全部 × zh/en
TTS_MODELS="melotts" bash tts/run.sh                 # 指定模型（可强制跑 enabled:false 的，如 kokoro）
TTS_DATASETS="aishell3" bash tts/run.sh              # 指定数据集
TTS_DATASETS="aishell3 librispeech zh_long" bash tts/run.sh   # 中英主集 + RTF 基准集
TTS_LIMIT=3 bash tts/run.sh                          # 冒烟（限条数）
TTS_SKIP_ANCHOR=1 bash tts/run.sh                    # 跳过 GT 锚点（已测过时）
```

> - 默认模型列表来自 config `models.tts` 中 **未标 `enabled: false`** 的项（当前 kokoro 已排除，中文合成偏短待修复）。
> - `librispeech` / `zh_hardcase` / `zh_long` **默认不在** `TTS_DATASETS` 中，需显式指定。
> - **要看可比的 RTF 必须跑 `zh_long`**（见 §4.7）。

流程（每数据集）：

1. **GT 锚点**（一次，幂等）：`asr_transcribe.py --scp <ds>/wav.scp` 转写**参考真人音频** → `eval_tts.py --gt_anchor` → `cer_gt`/`wer_gt` = **ASR 地板**
2. 逐模型：`synth_batch.py` 合成 → `benchmark/tts/syn/<model>_<ds>/`（记 RTF）
3. `asr_transcribe.py --wav_dir <syn>` 转写合成音 → `results/logs/<model>_<ds>_<asr>.hyp`
4. `eval_tts.py --asr_hyp` 算分 → 回环 CER/WER + S/D/I 分解 + 时长比 + 成功率 → `results/tts.csv`

> CSV 中 `GT` 行即锚点；回环值须**扣掉锚点**才能解读（见 §4.5）。
> 单独跑某一步也可以：`python tts/asr_transcribe.py --asr firered --wav_dir DIR --out h.hyp --lang zh`

**CosyVoice2 前置**（C++ LLM，需 tokenizer server 127.0.0.1:12345）：

```bash
cd CosyVoice2/scripts && nohup python cosyvoice2_tokenizer.py --host 127.0.0.1 --port 12345 &
# 板端建议拷贝模型到本地盘跑（NFS 链路慢）：COSYVOICE2_DIR=/root/cosyvoice2_local/CosyVoice2
```

**板端依赖**：评测侧需 `zhconv`（中文繁→简，缺失会告警并跳过导致 CER 偏高）；melotts（cn2an/inflect/pykakasi/pypinyin/g2p_en，**英文另需 NLTK 数据** `averaged_perceptron_tagger_eng` + `cmudict`）；kokoro（kokoro/misaki/num2words/phonemizer/ordered_set/mojimoji/jaconv + `pip install spacy && python -m spacy download en_core_web_sm` + `apt install espeak-ng`）；zipvoice（jieba，**英文见下方专用环境**）；cosyvoice2（`pip install --no-deps openai-whisper tiktoken`）。

### 3.1 zipvoice 英文需专用 python 环境

zipvoice 的 C++ 二进制在英文路径下会拉起 `cpp/scripts/py_daemon.py` 做 G2P，该脚本依赖
`piper_phonemize`（espeak-ng 的 Python 绑定）。**它只提供 cp39~cp312 的 wheel**，若板端
base 是 Python 3.13 则 `pip install` 会报 `no matching distribution`。中文走 pypinyin 查表，
不受影响。故英文需单独建一个 Python≤3.12 的环境：

```bash
conda create -n ZipVoice python=3.12 -y
conda activate ZipVoice
pip install piper_phonemize jieba pypinyin numpy soundfile
pip install axengine-x.x.x-py3-none-any.whl        # pyaxengine，按板端 release 装
```

然后在 `configs/benchmark.yaml` 填**环境名**（不要写绝对路径，路径由运行时 conda 位置推导）：

```yaml
paths:
  model_env:
    zipvoice: ZipVoice
```

合成时会打印 `子进程 python 环境 -> .../envs/ZipVoice/bin`。原理：二进制用
`execlp("python3", ...)` 拉起 daemon，走 PATH 解析，故 `synth_batch.py` 把该环境的
`bin/` 前置到子进程 PATH（只换解释器路径无效）。

### 3.2 zipvoice 目录完整性

官方两个分发渠道的文件集不同，需确认以下三项齐全，否则英文会失败：

| 需要 | 说明 |
|---|---|
| `cpp/scripts/py_daemon.py` | 英文 G2P daemon。**不在默认文件列表里**，缺失时报 `can't open file '.../cpp/scripts/py_daemon.py'` |
| `bin/zipvoice_ax650` | 下载后可能**无执行权限**，需 `chmod +x bin/zipvoice_*` |
| `resources/` `models/` | tokens.txt 与 axmodel |

## 4. 指标口径

### 4.1 中英分测原则（不可违背）

1. **中英永远分列**，不混算、不平均、不做任何换算 —— 两侧 ASR 不同、分母粒度不同，混算无意义。
2. **英文主指标 WER**（有空格天然分词，词定义无歧义），CER 作辅助诊断（专有名词/罕见词）。
3. **中文主指标 CER，绝不报 WER** —— 中文无词边界，分词器不同则 WER 不可比；汉字为最小语义单元，CER 稳定唯一。

### 4.2 评测 ASR 选型：板端单轨，中文用 FireRedASR-AED

**只测板端指标**，不引入 host 侧模型。评测 ASR 选**当前板端指标最好者**，让 ASR 地板尽量低：

| 语言 | ASR | 理由 |
|---|---|---|
| **中文** | **firered-aed** | `results/asr.csv` 中中文 CER 约为 sensevoice 的 1/10，**ASR 地板近乎归零 ⇒ TTS 回环 CER 几乎等于纯 TTS 误差** —— 这正是 TTS 评测最需要的性质。代价：RTF 约 7× 于 sensevoice（仅 python 无 C++ 二进制），但仍 <1，可全量跑 |
| **英文** | **待 GT 锚点实测定夺** | 候选：firered-aed（AED 架构，无需语言提示即可解码英文，BPE1000 覆盖中英）/ sensevoice（`--language` 真实生效）/ whisper-turbo（Whisper 系英文强项）。**`results/asr.csv` 目前无任何英文行 ⇒ 英文基线从未测定**，须以锚点最低者胜出 |

**⚠️ 已知阻断：vendor `test_wer.py` 出不了英文 WER**

`FireRedASR-AED/test_wer.py` 与 `SenseVoice/python/test_wer.py` 有相同两处硬伤：

| 问题 | 代码 | 后果 |
|---|---|---|
| GT 解析不支持空格 | `audio_path, gt = line.split(" ")`（无 maxsplit）| 英文转写含空格 → **ValueError: too many values to unpack（已实测确认）**。这就是英文回环从来跑不出结果的根因 |
| "WER" 实为字符级 | `min_distance()` 对字符串算编辑距离，分母 `len(reference)` | 中文正确（=CER）；**英文按字符算，标称 WER 实为 CER** |

**解法（已实现）**：转写与算分解耦 —— `tts/asr_transcribe.py` 只出 `hyp.txt`（`utt_id<TAB>文本`，TAB 分隔天然规避空格歧义），交给 `eval_tts.py --asr_hyp` → `tools/metrics_asr.py` 算分（中文按字 / **英文按词**）。一次改动即：绕开崩溃、英文得真 WER、归一化口径统一、**ASR 可配置切换**。

**已核实的非问题**：firered 的 `audio_dur=10` **不是截断上限**，而是内部 silero VAD 的分块大小 —— 长音频被切块后逐块解码再拼接（`_optimized_vad_split` + `_sequential_transcribe`），长句/hardcase 不会被截断。

**必须知道的局限**：
- ⚠️ **不可与论文数字横向比较**。Pocket TTS / Seed-TTS-Eval 用 Whisper-large-v3（英）/ Paraformer-zh（中）原始精度模型；本工程为板端量化口径。
- ⚠️ **ASR 自身误差仍混在回环指标里**（虽然 firered 已把中文地板压得很低），**须靠 GT 锚点（§4.5）扣除后解读**。
- ⚠️ **两套口径并存**：换 firered 前的 sensevoice 数值仍保留在 `results/tts.csv`（`note` 列标注 ASR 来源）。CSV 追加写，故同一 `(模型,数据集,指标)` 有多行时**表格取最后一行 = 最新口径**，`report.py` 会打印 `[info]` 提示被覆盖的旧值。**以精度更高的 firered 口径为准**，旧值仅作历史参考，两者不可直接比较。

**性能提示（板端本地盘）**：firered 的 axmodel 共 1.3 GB，NFS 链路约 2.7 MB/s，**冷加载需数分钟**。`configs/benchmark.yaml` 的 `paths.model_local_root`（默认 `/root/asr_local`）若存在同名模型子目录则优先使用本地盘副本：

```bash
# 板端一次性拷贝（约 1.3GB，注意本地盘余量）
mkdir -p /root/asr_local && cp -r /root/huyuan/workspace/8860_export/FireRedASR-AED /root/asr_local/
```
转写时会打印 `模型目录: ...（板端本地盘）` 或 `（NFS，冷加载较慢）`；host 侧该路径不存在会自动回退，两端可共用同一配置。

**不在板端范围内**（需 host GPU 大模型，本工程不做）：说话人相似度 SIM（WavLM-large）、神经 MOS（UTMOS）、论文级标准 ASR。

### 4.3 指标清单

| 指标 | 含义 | 成本 | 方向 | 状态 |
|---|---|---|---|---|
| `cer_loopback` / `wer_loopback` | 板端回环可懂度（合成音过板端 ASR vs 输入文本）| 已有 | 越低越好 | ✅ |
| `rtf` | **主口径：推理时间/音频时长，不含模型加载**（来源与陷阱见 §4.7）| 已有 | 越低越好 | ✅ |
| `rtf_e2e` | 逐条子进程墙钟，**含每条的模型加载**；仅参考，不与 `rtf` 混比 | 已有 | 越低越好 | ✅ |
| `audio_avg_s` | 合成音平均时长 —— **RTF 的必要上下文**（见 §4.7）| 零成本 | — | 📋 P1 |
| `mcd` | 梅尔倒谱距离（DTW 对齐）；**仅同参考说话人可比**，默认不作跨模型比较 | librosa | 越低越好 | ✅ 备用 |
| `cer_gt` / `wer_gt` | **GT 锚点**：真人参考音频过**同一板端 ASR** 的基线 | 零新增模型 | 参考基线 | 📋 P1 |
| `del_rate` / `ins_rate` / `sub_rate` | 错误类型分解：删除率 = 截断/漏读，插入率 = 重复/幻听，替换率 = 发音错 | 零成本 | 越低越好 | 📋 P1 |
| `len_ratio` | 合成音时长 / 参考音时长；显著 <1 = 截断，>>1 = 拖尾重复 | 零成本 | 接近 1 | 📋 P1 |
| `success_rate` | 合成成功条数 / 总条数（暴露超时/失败样本）| 零成本 | 越高越好 | 📋 P1 |

公式（编辑距离）：`WER = (S+D+I)_word / N_word`；`CER = (S+D+I)_char / N_char`。

> `del_rate`/`ins_rate`/`len_ratio` 的意义：总 CER 无法区分「截断漏读」与「重复幻听」这两种完全不同的故障；且 `len_ratio` 不依赖 ASR，是独立于 ASR 误差的故障探针。

### 4.4 文本归一化（直接影响指标数值）

| 语言 | 处理 |
|---|---|
| 中文 | **`zhconv` 繁→简** → 去中英标点 → 去空白 → 按字切分 |
| 英文 | 小写 → 去标点 → 数字读法与参考文本对齐 → **按空格切词**（真词级 WER）|
| 通用 | **剥离 ASR 富文本标签**（SenseVoice 会输出 `<\|zh\|>` `<\|HAPPY\|>` `<\|withitn\|>` 等语言/情感/事件标记；firered 不输出标签）|

> `tools/metrics_asr.py` 是归一化的**唯一事实来源**，不依赖 vendor `test_wer.py` 的内部口径（后者仅用于产出转写，见 §4.2）。
>
> 标签剥离为**防御性加固**：当前 SenseVoice 推理路径（`SenseVoiceAx.postprocess` 用 `ctc_logits[0, 4:]` 跳过前 4 帧）不输出标签，故历史中文数字未被标签污染；但换用会输出标签的路径（如 funasr `AutoModel`）时必需 —— 且必须**在去标点前**剥离，否则 `<|zh|>` 去掉 `<|>` 后残留 `zh` 成假词（实测 4 个标签会注入 18 个假字符）。

### 4.5 自检铁律

1. **GT 锚点必须先跑**：真人参考音频过与 TTS 回环**完全相同**的板端 ASR 管线，得 `cer_gt`/`wer_gt`。判读：
   - 中文锚点应与 `results/asr.csv` 中 **firered-aed 在 AISHELL-1 上的 CER 同量级**；显著偏高即**归一化有 bug，而不是 TTS 差**
   - 英文锚点为**首次测定**（现无英文基线），它同时决定英文用哪个 ASR、以及英文回环 WER 是否具备区分度
2. **回环值须扣锚点解读**：`回环值 ≈ 锚点` → 已触板端 ASR 上限，模型间不可再分辨；`回环值 ≫ 锚点` → 差值才是 TTS 真实损失。
3. **中英分列**：英文报 WER（CER 辅助），中文只报 CER，两侧不混算、不平均。
4. **不做同音字剔除**（保持简单可复现）：中文 CER 会把发音正确的同音字判为错误；部分论文注明"已剔除同音字"，口径不同不可直接对比。
5. **不与论文数字横向比较**：板端量化 ASR 口径（见 §4.2 局限）。

### 4.6 难例集（zh_hardcase）判读

| 类别 | 条数 | 看什么 |
|---|---|---|
| `hard_tw_*` 绕口令 | 40 | 连续相似音位的发音区分度；`sub_rate` 为主 |
| `hard_long_*` 长句 | 40 | 韵律稳定性与截断；配合 `del_rate` / `len_ratio` 看（<1 即截断）|
| `hard_num_*` 数字读法 | 35 | 前端文本规范化是否失效 |
| `hard_hom_*` 同音叠字 | 35 | 同音字/叠字鲁棒性 |

⚠️ **数字类是高方差诊断项，不宜作精确排名依据**：阿拉伯数字存在多种合法读法（`1200` 可读"一千二百"或"一二零零"；年份可读"二零二六"或"两千零二十六"），`text.ref` 只取最常见一种，故该类 CER 混入了"读法不同但不算错"的成分。用途是发现**明显失效**（数字被整段跳读、念成英文、或读出乱码），建议单独看该类，不与其他三类混算。

### 4.7 RTF 口径（两个坑，务必先读）

**坑一：逐条子进程的墙钟时间含模型加载，不能当主 RTF。**
`synth_batch.py` 每条起一个子进程，每次都重新加载模型（zipvoice 自报 `Engine load: 2340 ms` + `Vocoder load: 208 ms`）。该值写为 `rtf_e2e`，**不是**主口径。主口径 `rtf` 的来源按模型声明在 `tools/model_registry.py`：

| 模型 | 纯推理来源 | 说明 |
|---|---|---|
| melotts | `batch_driver: melotts_batch.py` | 载入一次 + warmup 跳首条，只计推理 |
| zipvoice | `rtf_pure_re` 解析二进制自报 | `NPU total` + `Vocoder total`（加载另行报告，天然排除）|
| cosyvoice2 | **无** | `main_ax650` 只打印 decode tokens，不自报耗时 ⇒ 只有 `rtf_e2e` |
| kokoro | — | 默认不跑（`enabled: false`，中文合成偏短待修复）|

**坑二（更隐蔽）：RTF 被测试集音频长度主导，短句会让 RTF 虚高数倍。**
axmodel 是**固定 shape**（zipvoice 为 `max_feat_len=1024`），每条推理耗时近似**恒定**（实测 `decoder(4 steps): 0.940 s`，与文本长短无关）。于是同一模型同一配置：

| 测试集 | 音频均长 | 纯推理/条 | 得到的 RTF |
|---|---|---|---|
| AISHELL-3 | 约 1.3s | 约 1.0s | **0.78** |
| ZipVoice 官方 README（中文句子）| 6.41s | 1.057s | **0.165** |

差 5 倍 —— **RTF 不是模型属性，而是「模型 × 音频长度」的函数**。故：

- 表格必看 `audio_avg_s` 列，脱离它谈 RTF 无意义；
- **跨模型 / 对官方可比的 RTF 一律看 `zh_long` 那一行**（40 条长句，定义见 `tools/common.py` 的 `RTF_REFERENCE_DATASET`），量级与各仓库官方口径一致；
- AISHELL-3 等短句集上的 RTF 只反映"短句场景的实际体验"，不可与官方值对比。

## 5. 实测结果

见顶层 README「已适配模型指标」自动表 / `results/summary.md`（`python tools/report.py` 生成，勿手写）。

## 6. 备注

- 零样本模型用各仓库自带固定 prompt（zipvoice: moss_prompts；cosyvoice2: prompt_files）
- 已知问题：kokoro 中文合成偏短/截断（回环 CER 高，模型质量问题；`del_rate`/`len_ratio` 落地后可量化确认）；zipvoice 二进制完成后不退出（synth_batch 已加超时容忍）；cosyvoice2 长文本慢路径会超时
- **英文侧合成覆盖不足**（`syn/` 下 EN 仅少量样本，`results/tts.csv` 无英文行），补英文 WER 前需先跑 `TTS_DATASETS=ljspeech bash tts/run.sh`
- 说话人相似度 SIM / 神经 MOS 需 host GPU 大模型，**不在板端 benchmark 范围内**
- 板端依赖详见 [audio-benchmark skill](.claude/skills/audio-benchmark/SKILL.md) 已知坑
- 数据与指标增强的完整修整计划见 [PLAN_objective_eval.md](PLAN_objective_eval.md)
