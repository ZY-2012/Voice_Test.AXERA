# TTS 客观评测增强 · 修整计划（板端 SenseVoice 单轨）

> 参照 `/data/shared/huyuan/8860_datasets/tts_eval_en_zh_separate.md` 的**中英分测原则**与**GT 锚点方法论**。
> **范围约定（已定）**：评测 ASR 统一用**板端 SenseVoice**，**只测板端指标**，不引入 host 侧标准 ASR / SIM / MOS 模型。
> 遵循 `CONTRIBUTING.md`：指标唯一事实来源 = `results/tts.csv` → `tools/report.py`；本文档不手写实测指标数字。

---

## 0. 范围裁剪说明（与参照 md 的差异）

参照 md 是论文复现口径（host 侧 Whisper-large-v3 / Paraformer-zh / WavLM）。本工程定位是**端侧 benchmark**，故只取其**方法论**，不取其模型：

| md 要素 | 本工程 | 说明 |
|---|---|---|
| 中英分测（EN→WER / ZH→CER） | ✅ **采纳** | 核心原则，指标结构已支持（`lang` 列） |
| GT 锚点（真人音频过同一 ASR） | ✅ **采纳，且更关键** | 板端量化 ASR 自身误差更大，不测锚点无法解读 TTS 数字 |
| 文本归一化（繁简/标点/数字） | ✅ **采纳** | 直接影响数值正确性 |
| 难例集 hardcase | ✅ **采纳** | 纯文本，板端可跑 |
| Whisper-large-v3 / Paraformer-zh | ❌ **不采纳** | 无 axmodel，板端跑不动；改用板端 SenseVoice |
| WavLM SIM / UTMOS MOS | ❌ **不采纳** | 需 host GPU 大模型，超出板端范围 |
| 与论文数字横向对比 | ❌ **不成立** | 板端量化 ASR 口径，**必须在 README 显式声明不可比** |

**代价与补偿**：失去 SIM/MOS 后，音质维度只剩 MCD。补偿方案是新增几个**零模型成本的板端客观指标**（S/D/I 分解、时长比、成功率），见 §2.2 —— 它们诊断力强且完全免费。

---

## 1. 现状差距（板端范围内）

| 项 | 现状 | 问题 |
|---|---|---|
| **英文评测** | `syn/` 下 EN 仅 kokoro 2 条；`results/tts.csv` **无任何 EN 行** | 英文侧事实上没测 |
| **ASR 英文基线** | `results/asr.csv` **8 个模型全是 aishell1/zh，无一条英文** | 板端 SenseVoice 的英文 WER 水平**完全未知**，EN 回环数字无从解读 |
| **GT 锚点** | 无 | 无法区分「TTS 差」与「ASR 差」——现有回环 CER 里混入了 SenseVoice 自身误差 |
| **归一化** | `metrics_asr.normalize` 仅 lower + 去标点 | 缺繁→简；未处理 SenseVoice 富文本标签与 ITN 数字 |
| **中文难例** | 无 | 日常句区分度不足 |
| **错误类型** | 只有总 CER | 不知是截断（删除多）还是重复（插入多）——kokoro 中文偏短问题无量化证据 |
| **合成成功率** | 仅打印，未入 CSV | cosyvoice2 有超时失败，成功率应作为指标 |

---

## 2. 测评方案

### 2.1 中英分测（核心原则）

| 语言 | 数据集 | 主指标 | ASR |
|---|---|---|---|
| **中文** | AISHELL-3、zh_hardcase | **`cer_loopback`**（绝不报 WER） | 板端 SenseVoice（`--language zh`） |
| **英文** | LJSpeech（+ 可选 LibriSpeech） | **`wer_loopback`** + `cer_loopback`（辅助诊断） | 板端 SenseVoice（`--language en`） |

两侧**永不混算、不平均**。现有 `tts/run.sh` 已按 `ds→lang` 分派，结构无需改。

### 2.2 指标清单

| 指标 | 含义 | 成本 | 状态 |
|---|---|---|---|
| `cer_loopback` / `wer_loopback` | 板端回环可懂度 | 已有 | ✅ |
| `rtf` | 推理时间/音频时长，不含加载 | 已有 | ✅ |
| `mcd` | 梅尔倒谱距离（DTW，仅同参考说话人可比） | librosa | ✅ 备用 |
| **`cer_gt` / `wer_gt`** | **GT 锚点**：真人参考音频过同一板端 SenseVoice | 零新增模型 | 📋 **P1 最高优先** |
| **`del_rate` / `ins_rate` / `sub_rate`** | 错误类型分解（删除/插入/替换率） | 零成本（编辑距离已算） | 📋 P1 |
| **`len_ratio`** | 合成音时长 / GT 音频时长 | 零成本（soundfile） | 📋 P1 |
| **`success_rate`** | 合成成功条数 / 总条数 | 零成本（synth_batch 已统计） | 📋 P1 |

**新增三个零成本指标的价值**（补偿失去的 SIM/MOS）：
- `del_rate` 高 = **截断/漏读**（kokoro 中文偏短的直接量化证据）；`ins_rate` 高 = **重复/幻听**；仅看总 CER 无法区分这两种完全不同的故障。
- `len_ratio` 显著 <1 = 合成被截断；>>1 = 拖尾/重复。**不依赖 ASR**，是独立于 ASR 误差的故障探针。
- `success_rate` 让 cosyvoice2 这类有超时失败的模型不再"隐藏"缺失样本（当前缺失条会被回环评测按空 hyp 计入或跳过，口径不透明）。

### 2.3 GT 锚点（P1 最高优先，必须先做）

**做法**：把评测集的**真人参考音频**（`benchmark/tts/<ds>/wav.scp` 指向的原始 wav）过与 TTS 回环**完全相同**的板端 SenseVoice 管线，以模型名 `GT` 写入 CSV。

**为什么在 SenseVoice 单轨下反而更关键**：
- 板端 SenseVoice 在 AISHELL-1 上的 CER 并非零（见顶层 README ASR 表），这部分误差**已经混在现有 TTS 回环 CER 里**。不知锚点 = 不知某个 TTS 数字里有多少是 ASR 的账。
- **英文锚点完全未知**（`asr.csv` 无英文行）。若 SenseVoice 英文 WER 本身偏高，则 EN 回环 WER 的区分度会被严重压缩——这个结论必须先测出来，才能判断 EN 回环 WER 是否可用、或需换更强的板端英文 ASR（如 whisper-turbo 已适配）。

**判读规则**：
- `TTS 回环值 ≈ 锚点` → TTS 已触板端 ASR 上限，模型间不可再分辨
- `TTS 回环值 ≫ 锚点` → 差值才是 TTS 真实损失
- `锚点异常高`（如 ZH 远超 asr.csv 的 aishell1 水平）→ **归一化有 bug**，不是数据/模型问题

**副产品**：这同时补上了 ASR 模块缺失的英文基线，`asr.csv` 与 `tts.csv` 可交叉印证。

### 2.4 文本归一化（板端口径）

| 语言 | 处理 |
|---|---|
| 中文 | `zhconv` 繁→简 → 去中英标点 → 去空白 → 按字切 |
| 英文 | 小写 → 去标点 → **数字读法统一**（LJSpeech 参考文本已是拼写形式，若 ASR 输出阿拉伯数字需转换对齐） |
| 通用 | **清理 SenseVoice 富文本标签**（`<|zh|>` `<|HAPPY|>` `<|withitn|>` 等语言/情感/事件标记必须剥离，否则计为错误） |

> `tools/metrics_asr.py` 现仅 lower + `[^\w\s]` 去标点：① 缺繁简；② SenseVoice 标签里的 `zh`/`HAPPY` 是 `\w` 字符，**去标点后残留成假词**，直接污染指标。需确认板端 SenseVoice `test_wer.py` 是否已剥离；若已剥离则沿用其口径，避免两套归一化打架。

---

## 3. 需新增的数据（对应问题 1）

| 数据 | 用途 | 规模 | 获取方式 | 优先级 |
|---|---|---|---|---|
| **无需新下载** —— 补跑现有 LJSpeech | 英文回环 WER（当前空缺） | 已有 200 条子集 | `TTS_DATASETS=ljspeech bash tts/run.sh` | **P1** |
| **zh_hardcase** | 中文难例，拉开区分度 | ≈150 句纯文本 | 自建 `tts/zh_hardcase/text`；四类：绕口令 / 长句>40字 / 数字日期读法 / 重复同音字 | P2 |
| **LibriSpeech test-clean**（可选） | 英文多说话人，替代单说话人 LJSpeech | 抽 200 条 | **复用 `asr/librispeech/`，零下载** | P2 |

**说明**：SenseVoice 单轨 + 只测板端 ⇒ 不再需要 `pairs.scp` 配对、WavLM 权重、Whisper/Paraformer 权重、24k 增强集、Seed-TTS-Eval 音频。**数据侧成本几乎为零**，主要工作在指标实现与归一化。

zh_hardcase 示例格式（`uid\t文本`）：
```
hard_tongue_001	四是四十是十十四是十四四十是四十
hard_long_001	在人工智能技术飞速发展的今天，语音合成系统已经能够生成接近真人水平的自然语音，但在处理长句时仍然面临韵律不自然的挑战
hard_num_001	二〇二六年八月三十一日，圆周率约等于三点一四一五九
hard_rep_001	红鲤鱼与绿鲤鱼与驴红鲤鱼与绿鲤鱼与驴
```

---

## 4. ASR 选型（对应问题 3）：中文换 **FireRedASR-AED**

`results/asr.csv` 中 firered-aed 的中文 CER 是 sensevoice 的**约 1/10**（0.575% vs 5.510%，见 CSV），把 ASR 地板压到近乎归零 —— **TTS 回环 CER 几乎等于纯 TTS 误差**，这正是 TTS 评测最需要的性质。故中文改用 firered-aed。

| 维度 | firered-aed | sensevoice | 结论 |
|---|---|---|---|
| 中文 CER（见 `asr.csv`）| **最优（约 0.6% 量级）** | 约 5.5% 量级 | **firered 胜**：地板近零 |
| RTF | 0.275（仅 python，无 C++ 二进制）| 0.039（C++）| firered 慢约 7×，但 <1 仍可全量跑 |
| 语言开关 | `--language` 参数**存在但代码未使用**（AED 架构无需语言提示，BPE1000 词表覆盖中英）| ✅ 真实生效（`model.infer(path, language)`）| 英文能力需实测验证 |
| 长音频 | `FireRedASRAxModel(..., audio_dur=10)` **硬编码 10s** | 需复核 | ⚠️ 长句/hardcase 有截断风险 |

### 4.1 ⚠️ 阻断问题：vendor `test_wer.py` 无法处理英文

审计 `FireRedASR-AED/test_wer.py` 与 `SenseVoice/python/test_wer.py`，**两者有相同的两处硬伤**：

| 问题 | 代码 | 后果 |
|---|---|---|
| **GT 解析不支持空格** | `audio_path, gt = line.split(" ")`（无 maxsplit）| 英文转写必然含空格 → **直接 ValueError 崩溃**。这极可能正是英文回环从未产出结果的根因 |
| **"WER" 实为字符级 CER** | `min_distance(reference, hypothesis)` 对**字符串**算编辑距离，`character_num = len(reference)` | 中文正确（=CER）；**英文按字符算，标称 WER 实为 CER**，命名误导 |

⇒ **无论换成 firered 还是留 sensevoice，沿用 vendor `test_wer.py` 都出不了英文 WER。**

### 4.2 解法：转写与算分解耦（架构小改，收益全面）

现状 `tts/run.sh` 是「跑 vendor `test_wer.py` → `grab_cer` 从 log 里 grep 数字」，**绕过了工程自己的 `tts/eval_tts.py`**。而 `eval_tts.py` 早已支持 `--asr_hyp`（`id\thyp` 文件）并调用 `tools/metrics_asr.py` —— 后者 `tokenize()` 是**中文按字 / 英文按词**，本来就能算真 WER。链路已铺好，只差接上：

```
新链路：  syn_dir/*.wav ─▶ tts/asr_transcribe.py（板端，ASR 可换）─▶ hyp.txt(id\thyp)
                                                                        │
                          text（参考）──────────────────────────────────┴─▶ eval_tts.py --asr_hyp
                                                                              └─▶ tools/metrics_asr.py
                                                                                  （中文按字 CER / 英文按词 WER）
```

**一次改动解决四个问题**：① 绕开 `split(" ")` 崩溃，英文可跑；② 英文得到**真正的词级 WER**；③ 归一化口径统一收归工程侧（繁简转换、SenseVoice 标签剥离、S/D/I 分解都在 `metrics_asr.py` 里加）；④ **换 ASR 只需换转写这一步**，firered / sensevoice / whisper-turbo 可配置切换、可交叉对比。

### 4.3 最终选型

| 语言 | 首选 ASR | 备选 | 说明 |
|---|---|---|---|
| **中文** | **firered-aed** | sensevoice | CER 地板近零，明确的赢 |
| **英文** | **firered-aed（待 GT 锚点验证）** | sensevoice（有真实语言开关）/ whisper-turbo（Whisper 系英文强项）| firered 是 AED 架构，无需语言提示即可解码英文；但**英文能力从未实测**（`asr.csv` 无任何英文行），须以 GT 锚点定夺 |

**决策规则**：先用 GT 锚点（真人音频过各候选 ASR）测出中英基线，**锚点最低者胜出**。这一步同时补上了 ASR 模块缺失的英文基线。

> `configs/benchmark.yaml` 建议改为分语言配置：`eval.tts.asr_model_zh: firered` / `asr_model_en: <锚点定夺>`（现为单一 `asr_model: sensevoice`）。

---

## 5. 文件改动清单

| 文件 | 改动 | 阶段 |
|---|---|---|
| `tts/README.md` | §1 数据集 / §4 指标口径与 ASR 选型说明 | ✅ **已完成** |
| `数据清单.md`（两份） | TTS 表登记规划数据 | ✅ **已完成** |
| `tts/PLAN_objective_eval.md` | 本文件 | ✅ **已完成** |
| `tts/asr_transcribe.py` | **新文件**：板端批量转写（`--asr firered\|sensevoice\|whisper-turbo` + `--wav_dir` → `hyp.txt`）。ASR 可换的关键 | **P1** |
| `tools/metrics_asr.py` | `normalize()` 增 `lang`：中文 `zhconv` 繁→简；剥离 ASR 富文本标签；`score()` 返回 `del/ins/sub` 分解 | **P1** |
| `tts/run.sh` | 回环改走 `asr_transcribe.py` + `eval_tts.py --asr_hyp`，**弃用 `grab_cer` 从 vendor log 抓数字**；每数据集先出 GT 锚点行（幂等） | **P1** |
| `tts/eval_tts.py` | 增 GT 锚点模式 + 写 `len_ratio`/`success_rate`（`--asr_hyp` 链路已有，无需重写） | **P1** |
| `tools/report.py` | `METRIC_FORMATS` 增 `cer_gt`/`wer_gt`/`del_rate`/`ins_rate`/`sub_rate`/`len_ratio`/`success_rate` | **P1** |
| `configs/benchmark.yaml` | `eval.tts` 改分语言 ASR（`asr_model_zh: firered` / `asr_model_en`）+ 补新指标名；`subset.tts` 增 `zh_hardcase` | **P1** |
| `tts/zh_hardcase/text` | 自建难例清单（+ `source.txt`） | P2 |
| `tools/make_subsets.py` | `sub_tts` 增 zh_hardcase（全量，无参考音频→跳过 wav.scp/MCD/锚点）+ 可选 librispeech | P2 |
| `tools/数据总结.md` | TTS 节补：板端单轨口径、firered 选型、不可与论文比、SIM/MOS 超出范围 | P1 |

**`report.primary_metric.tts`**：保持 `cer_loopback` 不变（板端主 KPI）。

**注意 hardcase 无参考音频** ⇒ 无 `wav.scp` ⇒ MCD 与 GT 锚点均不适用，`make_subsets`/`eval_tts` 需容忍缺失（现有代码已用 `if wav_scp.exists()` 保护，需复核 run.sh 的锚点分支同样容错）。

**注意 firered 的 `audio_dur=10` 硬编码**：长句/hardcase 若超 10s 可能被截断，导致删除率虚高。P1 需实测确认，必要时改参或分段。

---

## 6. 阶段划分

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0** | 文档层：本计划 + README ASR 选型/口径 + 数据清单 | ✅ 完成 |
| **P1 代码层** | ① `asr_transcribe.py` 解耦转写（firered/sensevoice 可切）② 归一化修正（繁简+标签剥离+S/D/I）③ GT 锚点模式 ④ 新指标（len_ratio/success_rate）⑤ run.sh 重接线 ⑥ report/config 注册 | ✅ 完成（host 侧已验证） |
| **P1 板端跑数** | GT 锚点定夺中英 ASR → 补跑英文全模型 → 中文用 firered 重跑 → `report.py --readme` | ⏳ **待板端执行**（`$BOARD_IP` 未设置，需用户提供或自行运行）|
| **P2 难例集** | zh_hardcase 150 句自建 + prepare/subset/text.ref 链路 | ✅ 完成（host 侧已验证）|
| **P2 英文多说话人** | LibriSpeech test-clean 2620 条/40 说话人（复用 `asr/librispeech`，零下载）| ✅ 完成（host 侧已验证）|
| **超出范围**（已记入 README）| SIM（WavLM）、MOS（UTMOS）、论文级标准 ASR —— 均需 host GPU 大模型 | — |

### 6.1 已实现文件（host 侧验证通过）

| 文件 | 状态 |
|---|---|
| `tools/metrics_asr.py` | ✅ 标签剥离（去标点前）+ zhconv 繁→简 + `align_ops()` S/D/I 分解；`load_text` 已用 maxsplit=1 |
| `tts/asr_transcribe.py` | ✅ 新增：`--asr firered\|sensevoice`、`--wav_dir` 或 `--scp`、`--ids` 对齐子集、模型载入一次 |
| `tts/eval_tts.py` | ✅ GT 锚点模式 + S/D/I + len_ratio + success_rate；MCD 改 `--mcd` 选择性开启 |
| `tts/run.sh` | ✅ 重接线：GT 锚点（幂等）→ 合成 → 转写 → 算分；弃用 vendor log 抓取 |
| `tools/report.py` | ✅ 新指标格式 + `Fmt.lower_better=None` 支持非单调指标（时长比不误标最优）|
| `configs/benchmark.yaml` | ✅ `eval.tts.asr_model_zh: firered` / `asr_model_en: sensevoice`（待锚点定夺）+ metrics 清单 |
| `tools/metrics_tts.py` | ✅ 修 MCD 量纲 bug（见 6.2）|
| `tts/zh_hardcase.tsv` | ✅ 新增：150 句难例定义（随仓库版本化，含数字类口语参考列）|
| `tts/librispeech_testclean_spk.txt` | ✅ 新增：40 个 test-clean 说话人 ID（含来源与交叉验证说明）|
| `tts/prepare_tts.py` | ✅ `prep_zh_hardcase()` + `prep_librispeech()`（按说话人过滤 test-clean）|
| `tools/make_subsets.py` | ✅ `sub_tts` 支持无参考音频数据集 + 透传 `text.ref` + 纳入 librispeech |

### 6.2 实施中发现并修正的问题

| 发现 | 判定 | 处理 |
|---|---|---|
| vendor `test_wer.py` 英文 GT 解析崩溃 | **已实测确认** `ValueError: too many values to unpack` | 解耦转写与算分，绕开 |
| vendor "WER" 英文实为字符级 | 确认（分母 `len(reference)`）| 改走 `metrics_asr` 词级切分 |
| **MCD 量纲错误（既存 bug）** | librosa MFCC 已是 dB 量纲，代码又乘 `10/ln10`，**放大约 4.34×**（自比对=0 说明算法本身对）| 系数改 `√2`；且 MCD 仅同说话人有意义，改默认关闭（`--mcd`）——**从未写入过 CSV，无历史数字受影响** |
| **CSV 速率量纲不一致** | 旧 run.sh 写百分数、旧 eval_tts.py 写 0~1 分数；report.py 按百分数渲染 | 统一为百分数（否则 12.37% 会渲染成 0.12%）|
| report.py 无法表达非单调指标 | 时长比 1.5（拖尾）会被误标"最优" | `Fmt.lower_better=None` 档位 |
| SenseVoice 是否输出富文本标签 | **不输出**（`postprocess` 用 `ctc_logits[0, 4:]` 跳过前 4 帧）⇒ 历史中文数字未被标签污染 | 标签剥离保留为防御性加固 |
| firered `audio_dur=10` 是否截断长音频 | **不截断**——是内部 silero VAD 的分块大小，切块后逐块解码再拼接 | 无需处理，撤回此前风险判断 |

### 6.3 首轮板端冒烟发现的 4 个 bug（已修并验证）

| # | Bug | 根因 | 修法 |
|---|---|---|---|
| ① | **GT 锚点静默跳过** | `wav.scp` 存的是制作时的 host 绝对路径，**板端实测解析 0/200**（经 `common.remap` 后 200/200）；且 `asr_transcribe` 无音频时 `return` → exit 0，`\|\|` 兜不住 | scp 路径过 `remap()`；无音频改 **exit 2**；run.sh 改 `if/then/else` 显式报警 |
| ② | `len_ratio` 无产出 | 同上（wav.scp 路径） | `eval_tts` 加 `_ref_path()` 走 remap；解析失败打 warn |
| ③ | `TTS_LIMIT` 对转写无效 | `$MLIM` 只传给了 synth_batch 与 GT 锚点，漏传模型转写 → 转了全部 200 条 | run.sh 补传；`--ids` 改按**行序**排序（与 `synth_batch` 的 `rows[:N]` 严格同口径，避免 glob 序错位）；`eval_tts` 加 `--limit` |
| ④ | **幂等重跑把 RTF 写成 0** | 全部命中跳过 → `infer_s=0` → `rtf=0/audio=0`，被当真值写入，**污染已有 RTF** | `synth_batch` 加 `n_new` 计数，全跳过时不改写 `_rtf.txt`；run.sh 对空/0/nan 拒写 |

验证输出（`TTS_LIMIT=5` 冒烟）：`GT锚点 CER=3.03%`、`时长比=1.083`、`转写 5 条`、`RTF=0.0000 无效（未重新合成），不写入 CSV`。

### 6.4 已确认的口径与性能决策

- **两套 ASR 口径并存**（用户决定）：sensevoice 旧值保留在 CSV，firered 为准。`report.py` 的 `pivot()` 取同 `(模型,数据集,指标)` 的**最后一行 = 最新口径**，并打印 `[info]` 提示被覆盖的旧值，避免静默误读。
- **模型走板端本地盘**（用户决定）：新增 config `paths.model_local_root`（默认 `/root/asr_local`），存在同名模型子目录则优先使用；host 侧不存在自动回退，两端共用同一配置。起因：firered axmodel 1.3GB + NFS 约 2.7MB/s ⇒ 冷加载数分钟，而 run.sh 每阶段独立加载（GT 锚点 1 次 + 每模型 1 次）。
  > ⚠️ 板端本地盘余量紧张：57G 中已用 91%，**仅剩 5.4G**；拷入 1.3GB 后约剩 4G。

---

## 7. 待办 / 待确认

**下一步（需板端）**：
```bash
# 板端 base 环境，冒烟先行
TTS_DATASETS=aishell3 TTS_MODELS=melotts TTS_LIMIT=3 bash tts/run.sh   # 验证链路
TTS_DATASETS=ljspeech TTS_MODELS=melotts TTS_LIMIT=3 bash tts/run.sh   # 验证英文（关键：此前必崩）
# 通过后全量；再 host 端 python tools/report.py --readme
```

1. **英文 ASR 由 GT 锚点定夺**：候选 firered / sensevoice / whisper-turbo，锚点最低者胜出（当前 config 暂填 sensevoice）。
2. **中文历史指标需重跑**：旧数字是 sensevoice 口径，换 firered 后不可混比；重跑后 `note` 列会自动带 `ASR=firered`。
3. **zh_hardcase（P2）**：≈150 句难例清单由我起草，还是你提供来源？
4. 板端需确认已装 `zhconv`（缺失会告警跳过繁→简，中文 CER 偏高）。
