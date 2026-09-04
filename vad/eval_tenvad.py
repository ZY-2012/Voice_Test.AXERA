#!/usr/bin/env python3
"""TEN VAD 在 benchmark 子集上评测（板端，C++ 可执行）。

TEN VAD 只提供 C API/可执行（ten-vad.axera 仓库），无 python 绑定，故用
`ten_vad_example in.wav out.wav` 逐条推理，解析 stdout 的每帧 `[i] prob, flag`
与自报 RTF（其计时不含 ten_vad_create/模型加载，与本工程 RTF 口径一致）。

帧长 = 256 采样（16ms，example.c 的 hop_size），故参考标签按 16ms 分帧；
silero 为 32ms —— 帧级 P/R/F1 对帧长不敏感，note 列记录实际帧长以便追溯。

用法: python eval_tenvad.py [--dataset ten_official librivad aishell1 picovoice]
环境: LIMIT=N 小样冒烟；PICO_SEC=600 picovoice 长流截前 N 秒（默认 600）
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, CmmDelta, remap
import metrics_vad as V

HOP = 256                      # example.c 固定帧移：256 采样 = 16ms @16k
FRAME_MS = HOP * 1000 // 16000
_PROB_RE = re.compile(r"^\[(\d+)\]\s+([0-9.]+),\s+(-?\d+)")
_RTF_RE = re.compile(r"Consuming time: ([0-9.]+)\(ms\), audio-time: ([0-9.]+)\(ms\)")


def tenvad_paths(cfg):
    """(可执行, 工作目录)：工作目录须含 axmodel/ten-vad-ax650.axmodel。"""
    code = Path(cfg["paths"]["model_root"]) / cfg["models"]["vad"]["tenvad"]["code_dir"]
    code = Path(remap(str(code)))
    return code / "install" / "example" / "ten_vad_example", code


def _run_env(workdir):
    env = os.environ.copy()
    libs = [str(workdir / "install" / "aarch64-lib" / "lib"), "/soc/lib", "/opt/lib"]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def _child_rss_mb(pid):
    """子进程峰值 RSS（MB）。读 VmHWM——内核自己维护的历史峰值，
    不会因轮询间隔错过瞬时高点；进程已退出返回 None。"""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


class _MemPeak(threading.Thread):
    """子进程运行期间轮询 NPU CMM 增量与子进程峰值 RSS。
    CMM 用 CmmDelta（峰值−基线）：/proc 里是全板共享计数，绝对值被其它子系统
    底噪主导（本板约 276MB），只有增量能归因到模型。
    RSS 不能用 resource.getrusage(RUSAGE_CHILDREN).ru_maxrss：它是"历来所有已
    回收子进程的最大值"，单调不减且混入 python 自身 fork 的开销（实测 11.8MB
    vs 真实 4.0MB），无法归因到当前数据集。"""

    def __init__(self, pid, cmm_probe=None, interval=0.02):
        super().__init__(daemon=True)
        # 注意别用 self._stop：Thread 内部已有同名方法，覆盖会让 join() 崩
        self.pid, self.interval = pid, interval
        self.cmm_probe = cmm_probe or CmmDelta()
        self.rss = None
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.cmm_probe.sample()
            r = _child_rss_mb(self.pid)
            if r is not None and (self.rss is None or r > self.rss):
                self.rss = r
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1)
        return self.cmm_probe.delta, self.rss


def infer_one(exe, workdir, wav_path, out_wav, sample_mem=False):
    """跑一条 → (probs, infer_sec, audio_sec, (cmm_delta, rss_peak))。失败返回 None。"""
    # CMM 基线必须在 fork 之前取：ten_vad_create 在进程启动后几毫秒就分配 NPU 内存，
    # 晚取基线会把模型自己的占用算进底噪，delta 偏小
    cmm_probe = CmmDelta() if sample_mem else None
    proc = subprocess.Popen([str(exe), str(wav_path), str(out_wav)], cwd=str(workdir),
                            env=_run_env(workdir), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    sampler = None
    if sample_mem:
        sampler = _MemPeak(proc.pid, cmm_probe=cmm_probe)
        sampler.start()
    try:
        out, _ = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    finally:
        mem = sampler.stop() if sampler else (None, None)
    probs, infer_ms, audio_ms = [], None, None
    for line in out.splitlines():
        m = _PROB_RE.match(line.strip())
        if m:
            probs.append(float(m.group(2)))
            continue
        m = _RTF_RE.search(line)
        if m:
            infer_ms, audio_ms = float(m.group(1)), float(m.group(2))
    if not probs or infer_ms is None:
        print(f"  [warn] 推理失败 {Path(wav_path).name}: {out.strip()[-300:]}")
        return None
    return np.asarray(probs), infer_ms / 1000.0, audio_ms / 1000.0, mem


def eval_uttlist(exe, workdir, scp, tmp_wav):
    """逐句集：返回 (指标, rtf, (cmm_peak, rss_peak))。内存只在第一条采样（每条都是
    同一个可执行 + 同一模型，占用一致；逐条起采样线程会拖慢整体）。"""
    lines = [l for l in Path(scp).read_text().splitlines() if l.strip()]
    lim = int(os.environ.get("LIMIT", "0"))
    if lim > 0:
        lines = lines[:lim]
    items, infer_s, audio_s, mem = [], 0.0, 0.0, (None, None)
    for i, line in enumerate(lines):
        _uid, wav_path, lab_path = line.split("\t")[:3]
        r = infer_one(exe, workdir, remap(wav_path), tmp_wav, sample_mem=(i == 0))
        if r is None:
            continue
        probs, t_in, t_au, peak = r
        infer_s += t_in
        audio_s += t_au
        if mem == (None, None):
            mem = peak
        ref = V.samples_to_frames(np.load(remap(lab_path)), 16000, FRAME_MS)
        n = min(len(probs), len(ref))
        if n:
            items.append((probs[:n], ref[:n]))
    if not items:
        return None, float("nan"), mem
    res = V.aggregate(items, ignore_label=None, threshold=0.5, frame_ms=FRAME_MS)
    return res, (infer_s / audio_s if audio_s else float("nan")), mem


def eval_picovoice(exe, workdir, info_txt, tmp_dir):
    """长流：整条 20h 无法一次载入（可执行会读全量到内存），截前 PICO_SEC 秒再跑。"""
    import soundfile as sf
    kv = dict(l.split("=", 1) for l in Path(info_txt).read_text().splitlines() if "=" in l)
    ref = np.loadtxt(remap(kv["labels"].strip()), dtype=int)   # 32ms 帧: 0=sil 1=unknown 2=speech
    cap = int(os.environ.get("PICO_SEC", "600"))
    sr = 16000
    wav, _ = sf.read(remap(kv["audio"].strip()),
                     frames=cap * sr if cap > 0 else -1, dtype="int16")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if cap > 0:
        ref = ref[:int(cap * 1000 / 32)]
    clip = tmp_dir / f"pico_{cap}s.wav"
    sf.write(clip, wav, sr, subtype="PCM_16")
    r = infer_one(exe, workdir, clip, tmp_dir / "pico_out.wav", sample_mem=True)
    clip.unlink(missing_ok=True)
    if r is None:
        return None, float("nan"), (None, None)
    probs, t_in, t_au, mem = r
    # 预测 16ms 帧 → 参考 32ms 帧栅格（重采样对齐，避免按帧数截断丢尾）
    pred = V.prob_to_frames(probs, len(ref))
    res = V.aggregate([(pred, ref)], ignore_label=1, threshold=0.5, frame_ms=32)
    return res, (t_in / t_au if t_au else float("nan")), mem


def scratch_dir():
    """临时 wav 目录：可执行必须写一个立体声输出（L=VAD flag），逐样本 fwrite。
    放 NFS 上会把评测拖成 I/O 瓶颈（板端↔服务器仅 ~2.7MB/s），故默认用本机盘。
    可用 TENVAD_SCRATCH 覆盖。"""
    d = Path(os.environ.get("TENVAD_SCRATCH") or (Path.home() / ".tenvad_scratch"))
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:                       # home 不可写时退回仓库内
        d = REPO / "results" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+",
                    default=["ten_official", "librivad", "aishell1", "picovoice"])
    args = ap.parse_args()
    exe, workdir = tenvad_paths(cfg)
    if not exe.exists():
        print(f"  未找到可执行 {exe}\n  先在 host 端跑: bash vad/build_tenvad.sh")
        return 1
    axm = workdir / "axmodel" / "ten-vad-ax650.axmodel"
    if not axm.exists():
        print(f"  缺 axmodel: {axm}（bash vad/build_tenvad.sh 会建软链，或跑 download_models.sh vad）")
        return 1

    dr = Path(cfg["paths"]["data_root"]) / "benchmark" / "vad"
    tmp_dir = scratch_dir()
    csv = REPO / "results" / "vad.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        csv.write_text("module,model,dataset,lang,metric,value,rtf,cmm_mb,os_mb,note\n")

    def rec(ds, res, rtf, mem):
        if res is None:
            print(f"  [{ds}] 无有效结果，跳过")
            return
        cmm, rss = mem
        lang = "zh" if ds == "aishell1" else "en"
        fms = 32 if ds == "picovoice" else FRAME_MS
        f_cmm = "" if cmm is None else f"{cmm:.1f}"
        f_rss = "" if rss is None else f"{rss:.1f}"
        with open(csv, "a") as f:
            for m in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
                f.write(f"vad,tenvad,{ds},{lang},{m},{res[m]:.4f},{rtf:.4f},"
                        f"{f_cmm},{f_rss},board frame={fms}ms\n")
        print(f"  [{ds}] F1={res['f1']:.3f} AUC={res['roc_auc']:.3f} "
              f"acc={res['accuracy']:.3f} RTF={rtf:.4f} RSS={f_rss or '—'}MB")

    for ds in args.dataset:
        if ds == "picovoice":
            info = dr / "picovoice" / "info.txt"
            if info.exists():
                rec(ds, *eval_picovoice(exe, workdir, info, tmp_dir))
        else:
            scp = dr / ds / "subset.scp"
            if scp.exists():
                rec(ds, *eval_uttlist(exe, workdir, scp, tmp_dir / "out.wav"))
            else:
                print(f"  [{ds}] 无子集 {scp}，跳过")
    (tmp_dir / "out.wav").unlink(missing_ok=True)
    (tmp_dir / "pico_out.wav").unlink(missing_ok=True)
    print(f">>> TEN VAD 评测完成 -> {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
