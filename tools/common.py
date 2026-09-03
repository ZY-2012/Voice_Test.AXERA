#!/usr/bin/env python3
"""公共工具：配置加载、确定性抽样、RTF/CMM/内存测量。
所有模块脚本共用，口径与 8860_export 的 asr-vad-summary skill 一致。"""
import hashlib
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_CFG = None


def load_config(path=None):
    """加载 configs/benchmark.yaml。无 PyYAML 时用极简解析兜底。
    路径自适应：host(/data/shared/huyuan) 与 板端(/root/huyuan/workspace) 为同一挂载，
    自动重映射；也可用环境变量 DATA_ROOT / MODEL_ROOT 覆盖。"""
    global _CFG
    if _CFG is not None:
        return _CFG
    cfg_path = Path(path) if path else REPO_ROOT / "configs" / "benchmark.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml
        _CFG = yaml.safe_load(text)
    except ImportError:
        _CFG = _mini_yaml(text)
    # 路径自适应
    p = _CFG.setdefault("paths", {})
    p["data_root"] = os.environ.get("DATA_ROOT", _remap_path(p.get("data_root", "")))
    p["model_root"] = os.environ.get("MODEL_ROOT", _remap_path(p.get("model_root", "")))
    return _CFG


def module_names(cfg=None):
    """模块清单（单一事实来源）：优先 config 的 modules: 顺序；缺失时从 models: 键派生。"""
    cfg = cfg or load_config()
    mods = cfg.get("modules")
    if isinstance(mods, dict) and mods:
        return list(mods.keys())
    return list((cfg.get("models") or {}).keys()) or ["asr", "vad", "se", "tts"]


def module_label(module, cfg=None):
    """模块展示名（config modules.<m>.label；缺失回退模块名大写）。"""
    cfg = cfg or load_config()
    node = (cfg.get("modules") or {}).get(module) or {}
    return node.get("label", module.upper())



# host 与 板端 的同一挂载前缀（本地 ↔ AX 板端）
_MOUNT_PAIRS = [("/data/shared/huyuan", "/root/huyuan/workspace")]

# TTS 数据集 -> 语言（单一事实来源：synth_batch / eval_tts / run.sh 都从这里派生，
# 避免"未列出即当英文"这类默认值 bug，如 zh_hardcase 被误判为 en）
TTS_DATASETS = {"aishell3": "zh", "zh_hardcase": "zh", "zh_long": "zh",
                "ljspeech": "en", "librispeech": "en"}

# RTF 主口径的测量集：固定 shape 的 axmodel 每条推理耗时近似恒定，RTF 被音频长短主导
# （AISHELL-3 均长仅约 1.3s，会让 RTF 虚高数倍，与各仓库用长句/段落报的官方值不可比）。
# 故跨模型/对官方可比的 RTF 一律看 zh_long（长句集）这一行。
RTF_REFERENCE_DATASET = "zh_long"


def tts_lang(dataset):
    """TTS 数据集的语言。未知数据集显式报错，不静默默认。"""
    try:
        return TTS_DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"未知 TTS 数据集 {dataset!r}；请在 tools/common.py 的 TTS_DATASETS 登记语言")


def _remap_path(pth):
    """若配置路径在本机不存在、但其挂载对端存在，则重映射（host↔board 透明切换）。"""
    if not pth:
        return pth
    if os.path.isdir(pth):
        return pth
    for a, b in _MOUNT_PAIRS:
        for src, dst in ((a, b), (b, a)):
            if pth.startswith(src):
                cand = pth.replace(src, dst, 1)
                if os.path.isdir(cand):
                    return cand
    return pth


def remap(pth):
    """翻译清单里的文件路径使同一清单在 host/board 两侧通用。
    按【路径实际是否存在】判断：原路径存在则用之；否则尝试挂载对端前缀，
    对端存在则用对端（不依赖根目录是否存在，兼容 /data/shared/huyuan 在板端为浅目录的情况）。"""
    if not pth or os.path.exists(pth):
        return pth
    for a, b in _MOUNT_PAIRS:
        for src, dst in ((a, b), (b, a)):
            if pth.startswith(src):
                cand = pth.replace(src, dst, 1)
                if os.path.exists(cand):
                    return cand
    return pth



def _mini_yaml(text):
    """仅支持本项目 yaml 子集（嵌套 dict + 标量 + 行内 [list]）的兜底解析。"""
    import ast
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            if val.startswith("{") and val.endswith("}"):
                # 行内 dict：手工转成 python 字面量
                inner = val[1:-1]
                d = {}
                for part in _split_top(inner):
                    k, _, v = part.partition(":")
                    d[k.strip()] = _scalar(v.strip())
                parent[key] = d
            elif val.startswith("[") and val.endswith("]"):
                parent[key] = [_scalar(x.strip()) for x in val[1:-1].split(",") if x.strip()]
            else:
                parent[key] = _scalar(val)
    return root


def _split_top(s):
    """按顶层逗号切分（忽略 [] 内的逗号）。"""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _scalar(v):
    v = v.strip().strip('"').strip("'")
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def deterministic_sample(items, n, seed):
    """确定性抽样：对每个 item 用 (seed, item) 哈希排序取前 n 个。
    同样的 items+n+seed 永远得到同样子集，且与列表原始顺序无关 → 可复现。"""
    if n is None or n <= 0 or n >= len(items):
        return list(items)

    def h(x):
        return hashlib.md5(f"{seed}:{x}".encode()).hexdigest()

    return sorted(items, key=h)[:n]


# ---------------- 板端资源测量 ----------------

def read_cmm_mb():
    """AX 板端 NPU CMM 当前占用（MB）。非板端返回 None。"""
    p = "/proc/ax_proc/mem_cmm_info"
    if not os.path.exists(p):
        return None
    try:
        for line in open(p):
            if "Cur" in line:
                for tok in line.replace(",", " ").split():
                    if tok.isdigit():
                        return int(tok) / 1024 / 1024
    except Exception:
        return None
    return None


def read_rss_mb():
    """当前进程 VmRSS（MB）。"""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


class RTFMeter:
    """RTF 计时器：累计推理耗时 / 音频总时长（不含加载，warmup 后调用）。"""

    def __init__(self):
        self.infer_sec = 0.0
        self.audio_sec = 0.0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self, audio_duration_sec):
        self.infer_sec += time.perf_counter() - self._t0
        self.audio_sec += audio_duration_sec

    @property
    def rtf(self):
        return self.infer_sec / self.audio_sec if self.audio_sec else float("nan")
