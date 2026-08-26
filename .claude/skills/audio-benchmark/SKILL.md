---
name: audio-benchmark
description: Axera 平台音频基准测试（ASR/VAD/SE/TTS 四模块）。当用户要求跑板端音频模型评测、补测/复测指标、维护 Voice_Test.AXERA benchmark 工程、下载/制作音频测试数据、排查板端推理问题时使用。
---

# Axera 音频基准测试（Voice_Test.AXERA）

四模块（ASR/VAD/SE/TTS）统一 benchmark 框架，目标：客户 5 步一键复现已适配模型指标。

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
bash run_benchmark.sh       # ⑤ 板端跑分 → results/summary.md
```
- 所有路径/参数在 `configs/benchmark.yaml`；脚本通过 `tools/cfg.py <key.path>` 读配置
- 小样冒烟：`LIMIT=5`（ASR/SE）、`TTS_LIMIT=3`、`PICO_SEC=600`（picovoice 长流截秒）
- 结果：`results/<module>.csv` → `tools/aggregate_results.py` → `summary.md`

## 关键约定（口径）

- **RTF = 推理时间/音频时长，不含模型加载/初始化**。C++ 可执行文件优先；C++ 值用 CLI 自报（whisper_cli/test_sensevoice 内部计时），python 值用载入一次 + warmup 1 条 + 计时 N 条的 `asr/rtf_probe.py` / `se/enhance_warm.py` / `tts/melotts_batch.py`
- ASR：中文 CER / 英文 WER，去标点转小写；子集同时提供 `wav/` 与 `aishell_S0764/`(软链) 两种音频目录名（各模型 test_wer 约定不同）
- TTS 主指标：**回环 CER**（合成音过 SenseVoice test_wer，输入文本作参考，`tools/metrics_tts.py` 口径）
- 子集抽样用 (seed,utt_id) 哈希确定性排序，可复现

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

## 常见任务 SOP

**补测某模型**：确认数据/模型就绪 → 板端 `DATASET=aishell1 ASR_MODELS='xxx' LIMIT=5 bash asr/run.sh` 冒烟 → 全量 → 更新 csv/README → host 端 aggregate + commit + push

**新增模型**：`tools/model_registry.py` 加 builder（cwd+argv）→ `configs/benchmark.yaml` 注册 dir/hf → 模块 run.sh 会自动纳入

**全量长任务**：板端 `setsid bash -c '...' &` 脱离 SSH（不挂本地轮询器，用户要进度时前台查）

## 当前实测基准（AX650N，中文 AISHELL-3）

- ASR CER(200条)：FireRedASR 0.57% < SenseVoice 5.51% ≈ WeNet 5.58% < Whisper-turbo 10.07% < Zipformer 13.22% < whisper small/base/tiny 15/22/43%
- VAD 帧级 F1：librivad 0.966 / picovoice 0.959 / aishell1 0.878；SileroVAD RTF 0.029
- SE (VoiceBank-DEMAND)：FastEnhancer PESQ 2.75/RTF 0.186 > GTCRN 2.55/0.249 > noisy基线 1.95
- TTS 回环 CER：MeloTTS 12.4% < ZipVoice 21.7% < CosyVoice2 29.3% < Kokoro 70.6%（kokoro 中文合成偏短，模型问题）
