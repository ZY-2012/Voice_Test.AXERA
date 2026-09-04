# 扩展指南（CONTRIBUTING）

Voice_Test.AXERA 面向持续新增：**数据集、模型、测试标准**都会不断扩充。本文件是扩展的唯一入口，改完这里说的步骤即可，公共脚本/文档**全部自动纳入，无需手改**。

## 核心约定（先读）

1. **指标唯一事实来源 = `results/*.csv`**（10 列表头：`module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note`）。改指标 = 改 CSV → 跑 `python tools/report.py --readme` 刷新顶层 README 与 `results/summary.md`。**任何文档不手写指标数字**。
2. **RTF 口径** = 推理时间 / 音频时长，**不含模型加载/初始化**。C++ 可执行优先；python 用"载入一次 + warmup 1 条 + 计时 N 条"探针。⚠️ 别在计时期间起后台采样线程（抢 GIL，实测 RTF 虚高 5%）。
3. **内存口径**：`cmm_mb` = NPU 内存**增量**（峰值 − 基线，`tools/common.CmmDelta`）——`/proc/ax_proc/mem_cmm_info` 的 `used=` 是全板共享计数，绝对值是别的子系统底噪；`os_mb` = 承载推理那个进程的 RSS，**跨语言不可比**（python 进程含 numpy/axengine 运行时，C++ 可执行只有几 MB），且要在「载入后、推理前」定格，否则量进评测脚本自己累积的数组。
4. 顶层 README 的指标表位于 `<!-- RESULTS:<module> -->…<!-- /RESULTS:<module> -->` 标记区块内，由 report.py 生成，勿手改。
5. 模块 README 统一 6 节骨架：1 数据集 / 2 制作 / 3 测试命令 / 4 指标口径 / 5 实测结果（指向自动表）/ 6 备注。
6. 所有路径相对数据根目录/仓库根；数据根目录由 `configs/benchmark.yaml` 的 `paths.data_root` 配置。

## 新增数据集（3 步）

1. `configs/benchmark.yaml`：在 `subset.<module>` 加抽样计数（如 `new_ds: 100`）。
2. 下载+制作：数据下载逻辑加进 `download_datasets.sh` 的对应 `dl_<module>()`；制作逻辑加进模块 prepare 脚本或 `tools/make_subsets.py` 的对应 `sub_<module>()`。
3. `数据清单.md` 对应模块表加一行（名称/数量/下载地址）。

## 新增模型（2 步）

1. **SE/TTS**：`tools/model_registry.py` 加 builder 函数（返回 `(workdir, argv)`，签名 `def xxx(model_dir, chip, *args) -> tuple[Path, list[str]]`）+ `REGISTRY` 加条目（kind/sr/type/langs/builder/out_glob 等）。
   **ASR**：只需 config（见下），asr/run.sh 从 config 派生模型列表。
   **VAD**：加 `vad/eval_<model>.py`（产出每帧语音概率 → `tools/metrics_vad.py` 聚合 → 追加 10 列 `vad.csv`），并在 `vad/run.sh` 里加一段分支；模型清单由 config `models.vad` 派生（`VAD_MODELS` 可覆盖）。若模型只有 C++ 可执行（如 tenvad），config 额外写 `code_dir:` 指向推理代码仓库，交叉编译脚本参考 `vad/build_tenvad.sh`。
2. `configs/benchmark.yaml` 的 `models.<module>.<name>` 加 `{hf: AXERA-TECH/xxx, dir: 本地目录名}`。

然后：`LIMIT=3 bash <module>/run.sh` 冒烟 → 全量 → `python tools/report.py --readme` 刷新并提交。

## 新增模块（2 步）

1. `configs/benchmark.yaml`：`modules:` 加一行（`label` + `scope` 口径模板）+ `models:`/`eval:`/`subset:` 各加一节。
2. 建 `<module>/` 目录：必需 `run.sh`（板端评测，写 10 列 `<module>.csv`）+ `README.md`（6 节模板）；按需 `prepare*.py`/`eval*.py`。

之后 `download_datasets.sh`/`prepare_datasets.sh`/`run_benchmark.sh`（自动发现 run.sh）/`tools/make_subsets.py`/`tools/report.py`（汇总、README 注入、状态矩阵）全部自动纳入。

## 新增测试标准（指标）

1. `tools/metrics_<module>.py` 加计算函数（或新建 metrics 文件）。
2. 模块 run.sh 或 eval 脚本把新指标写入 `<module>.csv`（metric 列用新名字）。
3. 如需显示格式：`tools/report.py` 的 `METRIC_FORMATS` 加一行（label/小数位/单位/越低越好）；如需作为排序主指标：`configs/benchmark.yaml` 的 `report.primary_metric` 更新。

## 提交清单

- 指标有变化：先 `python tools/report.py --readme`（生成 summary + 刷新 README）再 commit。
- 检查无模型/缓存产物混入（`git status`；数据、模型、results/ 均 gitignore）。
- 板端 root 生成的文件记得 chown 回仓库属主。
