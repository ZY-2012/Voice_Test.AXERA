---
name: audio-benchmark
description: Axera 平台音频基准测试（ASR/VAD/SE/TTS 四模块）。当用户要求跑板端音频模型评测、补测/复测指标、维护 Voice_Test.AXERA benchmark 工程、下载/制作音频测试数据、排查板端推理问题时使用。
---

# Axera 音频基准测试（Voice_Test.AXERA）

四模块（ASR/VAD/SE/TTS）统一 benchmark 框架，目标：使用者 5 步一键复现已适配模型指标。

## 目录与主机

- **工程仓库**：`/data/shared/huyuan/8860_VoiceTest/Voice_Test.AXERA`（GitHub: ZY-2012/Voice_Test.AXERA，main 分支直接推送）
- **数据根**：`/data/shared/huyuan/8860_datasets`（asr/vad/se/tts + benchmark/ 子集）
- **模型根**：`/data/shared/huyuan/8860_export`（各模型推理仓库 + axmodel + asr_vad_summary.md 历史指标）
- **板端**：AX650N `root@<板端IP>`（密码用环境变量 $BOARD_PASS 注入，不落库）。本地 `/data/shared/huyuan/` = 板端 `/root/huyuan/workspace/`（**同一 NFS 挂载**，本地改文件板端立即可见）
- 板端命令模板：
  ```bash
  timeout 60 sshpass -p "$BOARD_PASS" ssh -o StrictHostKeyChecking=no root@$BOARD_IP \
    "source /root/miniforge3/etc/profile.d/conda.sh && conda activate base && <命令>"
  ```

## 五步复现流程

```bash
bash setup_env.sh           # ① 依赖（板端 --board；板端补: pesq pystoi zhconv openai-whisper tiktoken misaki num2words phonemizer jieba cn2an inflect pykakasi pypinyin g2p_en + apt espeak-ng）
bash download_datasets.sh   # ② 下原始数据（asr/vad/se/tts 可选模块）
bash prepare_datasets.sh    # ③ 制作测试集 + 确定性抽标准子集（FULL=1 全量）
bash download_models.sh     # ④ 从 HF AXERA-TECH 拉 axmodel
bash run_benchmark.sh       # ⑤ 板端跑分 → results/*.csv（末尾自动跑 report.py 生成 summary.md）
```
- 指标刷新（提交前必做）：`python tools/report.py --readme` 把 CSV 指标注入顶层 README 标记区块（--dry-run 预览；--only <module> 单模块调试）

- 所有路径/参数在 `configs/benchmark.yaml`；脚本通过 `tools/cfg.py <key.path>` 读配置
- 小样冒烟：`LIMIT=5`（ASR/SE）、`TTS_LIMIT=3`、`PICO_SEC=600`（picovoice 长流截秒）
- 结果：`results/<module>.csv` → `python tools/report.py` → `summary.md`；**提交前必跑 `python tools/report.py --readme`** 刷新顶层 README 自动指标表。README 标记区块须成对（`<!-- RESULTS:<m> -->…<!-- /RESULTS:<m> -->`、`<!-- STATUS-MATRIX -->…<!-- /STATUS-MATRIX -->`），report 注入幂等

## 关键约定（口径）

- **RTF = 推理时间/音频时长，不含模型加载/初始化**。C++ 可执行文件优先；C++ 值用 CLI 自报（whisper_cli/test_sensevoice 内部计时），python 值用载入一次 + warmup 1 条 + 计时 N 条的 `asr/rtf_probe.py` / `se/enhance_warm.py` / `tts/melotts_batch.py`
- ASR：中文 CER / 英文 WER，去标点转小写；子集同时提供 `wav/` 与 `aishell_S0764/`(软链) 两种音频目录名（各模型 test_wer 约定不同）
- TTS 主指标：**回环 CER**（合成音过 SenseVoice test_wer，输入文本作参考，`tools/metrics_tts.py` 口径）
- 子集抽样用 (seed,utt_id) 哈希确定性排序，可复现
- **模块/模型清单单一事实来源**：config `modules:`/`models:` 节；shell 用 `tools/cfg.py modules`（dict→空格分隔键）取清单，python 用 `common.module_names()`；新增模型=registry+config 两步（`tools/model_registry.py` + `configs/benchmark.yaml`），新增模块=config+建目录两步，详见 `CONTRIBUTING.md`

## 已知坑（重要，踩过并修复）

1. **NFS 链路仅 2.7MB/s**（板端↔服务器，服务器本地 4.8GB/s）：重模型（如 cosyvoice2）拷板端本地盘 `/root/cosyvoice2_local/` 跑，`COSYVOICE2_DIR` 环境变量覆盖
2. **NFS 频繁断连**：重挂 `mount -t nfs -o nolock $NFS_SERVER:/data/shared/huyuan/ /root/huyuan/workspace`；已加 fstab + x-systemd.automount。注意板端 `/data/shared/huyuan` 是浅目录，路径重映射按"文件是否存在"判断（`tools/common.py` remap）
3. **cosyvoice2 tokenizer server**（127.0.0.1:12345，main_ax650 依赖）：`cd CosyVoice2/scripts && nohup python cosyvoice2_tokenizer.py --host 127.0.0.1 --port 12345`；板端重启/异常后要手动拉起（systemd 常驻化待用户批准）
4. **subprocess PIPE 死锁**：日志量大的二进制（zipvoice）不能用 capture_output；统一重定向到 `_run.log`
5. **部分二进制 "Done!" 后不退出**（zipvoice）：subprocess 加 `--timeout`，输出已生成即算成功
6. **main_ax650 的 output*.wav 残留**：跑前清理 + 按 mtime 取最新（否则全取到旧文件）
7. **pkill -f 模式会匹配自身命令行导致自杀**：模式避免出现在命令文本里，或分两步执行
8. **vendor 兼容补丁已打**（改的是 8860_export 下文件，记录在案）：
   - Whisper/python/test_wer.py：UTF-8 解码异常 try/except 跳过坏样本
   - kokoro.axera/inference_utils.py：misaki 0.7.4 API 漂移（ZHG2P 无参 + str/tuple 返回）
   - 板端 WeNet units.txt 已是 AIShell-1 版（4233 词），勿换
9. **git 操作只在 host 端做**（板端 root 对 NFS 仓库 dubious ownership）；提交前确认无模型/缓存产物混入（如 melotts 的 bert-* 目录，已 gitignore）；推送用 SSH remote `git@github.com:ZY-2012/Voice_Test.AXERA.git`（HTTPS 凭据在 VS Code askpass 不可用）
10. pkill 大批进程后 NPU 可能瞬时异常（AX_ENGINE_CreateHandle failed），重试一次通常恢复
11. **results/ 与 tools/ 属主**：板端 root 跑完会写 root 属主文件，host 端再跑脚本可能 PermissionError——板端 `chown -R 1055:1001 <repo>/results <repo>/tools` 一次恢复；`__pycache__` 同理（勿在 NFS 上混用两端 python）
12. **README 注入标记**：只写开始标记不写结束标记时 report.py 会告警跳过（`<!-- RESULTS:x --><!-- /RESULTS:x -->` 连写即可）
13. **scp/清单里的绝对路径必须过 `common.remap()`**：`wav.scp` 等清单是 host 侧制作的，存的是 `/data/shared/huyuan/...`；板端该前缀是浅目录，**直接解析 0/200**，经 remap 后 200/200。凡是读清单里路径再 `open`/`exists` 的新代码都要先 remap（TTS 的 GT 锚点与 len_ratio 曾因此静默失效）
14. **大模型走板端本地盘**：config `paths.model_local_root`（`~/asr_local`，支持 ~ 展开），存在同名子目录即优先使用。firered axmodel 1.3GB + NFS 约 2.7MB/s ⇒ 冷加载数分钟。板端本地盘余量紧张（57G 已用 91%，剩约 4G），拷前先 `df -h /`
15. **幂等跳过时别写 RTF**：合成全部命中已存在会得到 `infer_s=0` → `RTF=0`，写入 CSV 会污染已有指标。`synth_batch.py` 用 `n_new` 计数判断，run.sh 对空/0/nan 拒写
16. **vendor `test_wer.py` 处理不了英文**：`audio_path, gt = line.split(" ")` 无 maxsplit，英文转写含空格必崩；且其 "WER" 是对字符串算编辑距离（英文实为 CER）。TTS 已改为「`asr_transcribe.py` 只出转写 + `tools/metrics_asr.py` 统一算分」，勿再退回抓 vendor log
17. **ZipVoice 两份副本各缺一半**：`ZipVoice.AXERA`(ModelScope) 有 `cpp/` 但 bin/resources 空；`ZipVoice.AXERA_HUG`(HF) 齐全但无 `cpp/`。现 config 指向 `_HUG`，并把 `cpp` 软链到 MS 副本；HF 的 `bin/*` 下载后**无执行权限**，需 `chmod +x`
18. **ZipVoice 英文需专用 python 环境**：`cpp/scripts/py_daemon.py`（英文 G2P daemon）依赖 `piper_phonemize`，它只出 cp39~cp312 wheel，板端 base 是 3.13 装不上。板端已有 `ZipVoice` env（3.10，含该包）；config `paths.model_env.zipvoice` 填**环境名**即可（路径由运行时 conda 位置推导，勿写绝对路径）。原理：二进制用 `execlp("python3")` 走 PATH，故 synth_batch 前置该 env 的 bin/ 到子进程 PATH。中文走 pypinyin 查表不受影响
19. **melotts 的 `melotts_batch.py` 必须先 chdir 到 MeloTTS/python**：vendor 用相对路径加载 BERT（`model_id='bert-base-multilingual-uncased'` 是同目录下的文件夹）。`synth_batch` 走子进程 `cwd=MODEL_DIR` 所以无此问题
20. **melotts 英文需 NLTK 数据**（`averaged_perceptron_tagger_eng` + `cmudict`，g2p_en 词性标注用）。板端直连 nltk 服务器会挂住并留下 `*.zip.lock`（残留锁阻塞重试，需先删）。**不用往板端拷**：数据放共享盘 `model_root/_nltk_data`（host 侧 `nltk.download(..., download_dir=...)` 下一次），代码自动注入 `NLTK_DATA`
21. **NPU 偶发 `Failed to initialize ax sys engine.`**：与坑 10 同源，重试即恢复；曾导致 melotts 英文与 GT 锚点整段静默失败，排查时先重试再怀疑代码
22. **C++ demo 的临时输出别落 NFS**：TEN VAD 的 `ten_vad_example` 必须写一路立体声 wav（逐样本 fwrite）。放共享盘时 200 条 librivad 跑 30 分钟还没完（子进程长期 D 状态），换本机盘 `~/.tenvad_scratch` 后约 17 分钟。凡 vendor 可执行强制写输出的，一律指到本机盘
23. **子进程峰值内存读 `/proc/<pid>/status` 的 `VmHWM`**：`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` 是"历来所有已回收子进程的最大值"，单调不减且混入 python 自身 fork 开销（实测 11.8MB vs 真实 4.0MB）；轮询 `VmRSS` 又会漏掉瞬时峰值，`VmHWM` 是内核维护的峰值，随时读都准

## 常见任务 SOP

**补测某模型**：确认数据/模型就绪 → 板端 `DATASET=aishell1 ASR_MODELS='xxx' LIMIT=5 bash asr/run.sh` 冒烟 → 全量 → host 端 `python tools/report.py --readme`（生成 summary + 刷新 README）→ commit + push

**新增模型**：`tools/model_registry.py` 加 builder（SE/TTS；ASR 只需 config）→ `configs/benchmark.yaml` 的 `models.<module>` 注册 `{hf,dir}` → 模块 run.sh 自动纳入 → 冒烟/全量 → `report.py --readme`。新增数据集/模块/测试标准步骤见 `CONTRIBUTING.md`

**全量长任务**：板端 `setsid bash -c '...' &` 脱离 SSH（不挂本地轮询器，用户要进度时前台查）

## 指标唯一事实来源

指标一律以 `results/*.csv` 经 `python tools/report.py` 生成为准（顶层 README 自动表 / results/summary.md）。任何文档不手写指标数字；扩展步骤见 `CONTRIBUTING.md`（新增数据集/模型/模块均为 config+目录两步）。
