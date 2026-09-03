#!/usr/bin/env python3
"""TTS 批量合成（板端）：把 benchmark TTS 子集的 text 逐句合成 -> syn_dir/<id>.wav。
每句调用模型仓库自带 demo/binary（命令见 tools/model_registry.py），记录端到端 RTF。
用法: python synth_batch.py --model kokoro|melotts|zipvoice|cosyvoice2 --dataset aishell3|ljspeech [--chip ax650]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, TTS_DATASETS, tts_lang
from model_registry import REGISTRY


def _parse_pure_infer_s(spec, log_path, offset=0):
    """从 _run.log 解析各条的**纯推理**耗时并求和（秒）。
    依赖二进制自报计时（registry 的 rtf_pure_re），其加载耗时另行报告故天然排除。
    offset：日志为追加模式，只解析本次运行新增的部分，避免累加历次运行。
    无正则或无匹配 -> None（该模型拿不到纯 RTF）。"""
    import re
    pats = spec.get("rtf_pure_re")
    if not pats or not Path(log_path).exists():
        return None
    with open(log_path, errors="ignore") as f:
        f.seek(offset)
        text = f.read()
    total = 0.0
    hit = False
    for p in pats:
        vals = [float(x) for x in re.findall(p, text)]
        if vals:
            hit = True
            total += sum(vals)
    return total if hit else None


def _resolve_env_bin(value):
    """把 config 的 model_env 值解析成 bin 目录。

    值可以是**环境名**（推荐，可移植：如 `ZipVoice`）——此时从当前运行的 conda 安装位置
    推导 `<conda_root>/envs/<name>/bin`，故各人 miniforge/anaconda 装在哪都能用；
    也可以是绝对/相对路径（含路径分隔符时按路径处理），供非 conda 场景。
    """
    import os
    import sys
    if not value:
        return None
    if os.sep in value or value.startswith("~"):
        return Path(os.path.expanduser(value))
    # 环境名：从 CONDA_PREFIX / sys.prefix 反推 conda 根目录（去掉可能的 envs/<x> 后缀）
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    root = Path(prefix)
    if root.parent.name == "envs":
        root = root.parent.parent
    return root / "envs" / value / "bin"


def _build_sub_env(cfg, model):
    """子进程环境：若 config 为该模型配了专用 python 环境，把其 bin/ 前置到 PATH。

    配置项 `paths.model_env.<model>`，值填**环境名**（可移植，勿写绝对路径）：
        model_env:
          zipvoice: ZipVoice
    为何改 PATH 而不是只换解释器：zipvoice 的 C++ 二进制用 `execlp("python3", ...)`
    拉起英文 G2P daemon，走 PATH 解析；只传解释器路径无效。
    未配置或目录不存在 -> 用当前环境（并给出提示）。
    """
    import os
    env = dict(os.environ)
    # NLTK 数据随模型分发（见 melotts_batch.py 注释）：英文 g2p_en 词性标注需要，
    # 板端直连下载常挂住。放在 model_root/_nltk_data，此处注入子进程。
    nltk_dir = Path(cfg["paths"]["model_root"]) / "_nltk_data"
    if nltk_dir.is_dir():
        env["NLTK_DATA"] = (f"{nltk_dir}{os.pathsep}{env['NLTK_DATA']}"
                            if env.get("NLTK_DATA") else str(nltk_dir))
    conf = (cfg.get("paths", {}).get("model_env") or {})
    bin_dir = _resolve_env_bin((conf.get(model) or "").strip())
    if bin_dir is None:
        return env, None
    if not bin_dir.is_dir():
        return env, f"[warn] {model} 的专用 python 环境不存在: {bin_dir}（回退当前环境，可能缺依赖）"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(bin_dir.parent)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env, f"子进程 python 环境 -> {bin_dir}"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[k for k, v in REGISTRY.items() if v["kind"] == "tts"])
    ap.add_argument("--dataset", required=True, choices=sorted(TTS_DATASETS))
    ap.add_argument("--chip", default=cfg["models"]["chip"])
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只合成前 N 条（重模型如 cosyvoice2 用）")
    ap.add_argument("--timeout", type=int, default=300, help="单条子进程超时秒数（部分二进制完成后不退出）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已存在的 wav 强制重新合成（口径/参数变更后作废旧产物时用）")
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    lang = tts_lang(args.dataset)
    if lang not in spec.get("langs", ["zh", "en"]):
        print(f"  {args.model} 不支持 {lang}，跳过"); return
    model_dir = Path(cfg["paths"]["model_root"]) / cfg["models"]["tts"][args.model]["dir"]
    text_f = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset / "text"
    if not text_f.exists():
        print(f"无子集 {text_f}"); return
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / "syn" / f"{args.model}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [l.split("\t", 1) for l in text_f.read_text(encoding="utf-8").splitlines() if "\t" in l]
    if args.limit > 0:
        rows = rows[:args.limit]
    infer_s = 0.0
    audio_new_s = 0.0   # **仅本次新合成**条目的音频时长（RTF 分母必须与 infer_s 同集合，
                        # 否则幂等跳过时 infer(新) / audio(全) 会低估 RTF）
    ok = 0
    n_new = 0          # 本次实际合成条数（幂等跳过的不算）
    # 日志重定向到文件而非 capture_output：部分模型(如zipvoice)日志量大，
    # PIPE 缓冲写满会导致子进程阻塞死锁
    logf = (out_dir / "_run.log").open("a")
    log_start = logf.tell()   # 追加模式：只解析本次新增日志，避免累加历次运行的自报耗时
    sub_env, env_note = _build_sub_env(cfg, args.model)
    if env_note:
        print(f"  {env_note}")
    for uid, text in rows:
        outp = out_dir / f"{uid}.wav"
        if not args.fresh and outp.exists() and outp.stat().st_size > 0:
            ok += 1
            continue  # 已合成跳过（幂等重跑）；--fresh 时覆盖重合成
        workdir, argv = spec["builder"](model_dir, args.chip, lang, text, outp.resolve())
        # 输出 output*.wav 的模型(如 cosyvoice2)：先清旧文件，否则会取到上次残留
        if spec.get("out_glob"):
            for f in Path(workdir).glob(spec["out_glob"]):
                f.unlink()
        t0 = time.perf_counter()
        try:
            r = subprocess.run(argv, cwd=workdir, stdout=logf, stderr=subprocess.STDOUT,
                               timeout=args.timeout, env=sub_env)
        except subprocess.TimeoutExpired:
            r = None  # 部分二进制(如 zipvoice)完成合成后不退出，超时即可，outp 已生成
        infer_s += time.perf_counter() - t0
        # cosyvoice2 输出 output*.wav，需改名到 outp
        if spec.get("out_glob") and not outp.exists():
            cand = sorted(Path(workdir).glob(spec["out_glob"]), key=lambda p: p.stat().st_mtime)
            if cand:
                shutil.move(str(cand[-1]), outp)
        if outp.exists():
            try:
                audio_new_s += sf.info(outp).duration
            except Exception:
                pass
            ok += 1
            n_new += 1
        elif ok == 0 and (r is None or r.returncode != 0):
            logf.flush()
            tail = Path(logf.name).read_text(errors="ignore")[-400:]
            print(f"  [{args.model}] 首条失败: {' '.join(str(a) for a in argv[:4])}...\n  log: {tail}")
    logf.close()
    rtf_f = out_dir / "_rtf.txt"
    pure_f = out_dir / "_rtf_pure.txt"
    if n_new == 0:
        # 全部命中幂等跳过：infer_s=0 会算出 RTF=0，写入会污染已有指标 —— 保留旧值
        old = rtf_f.read_text().strip() if rtf_f.exists() else ""
        print(f">>> {args.model}/{args.dataset}: 全部已存在({ok}/{len(rows)})，未重新合成 -> {out_dir}"
              f"  RTF(e2e) 沿用旧值={old or 'NA'}")
        # 纯 RTF 仍可从**已有**日志补算（无需重新合成）：日志与目录里的 wav 一一对应
        if not pure_f.exists():
            pure = _parse_pure_infer_s(spec, out_dir / "_run.log")   # offset=0：解析全部历史
            tot_audio = 0.0
            for uid, _ in rows:
                w = out_dir / f"{uid}.wav"
                if w.exists():
                    try:
                        tot_audio += sf.info(w).duration
                    except Exception:
                        pass
            if pure is not None and tot_audio:
                print(f"    从已有日志补算纯推理 RTF(不含加载)={pure / tot_audio:.4f}")
                pure_f.write_text(f"{pure / tot_audio:.4f}\n")
    else:
        rtf = infer_s / audio_new_s if audio_new_s else float("nan")
        print(f">>> {args.model}/{args.dataset}: 合成 {ok}/{len(rows)} 条（本次新增 {n_new}）"
              f" -> {out_dir}  RTF(e2e,含加载)={rtf:.4f}")
        rtf_f.write_text(f"{rtf:.4f}\n")
        # 纯推理 RTF（主口径，不含模型加载）：解析二进制自报计时（只看本次新增日志）
        pure = _parse_pure_infer_s(spec, out_dir / "_run.log", log_start)
        if pure is not None and audio_new_s:
            print(f"    纯推理 RTF(不含加载)={pure / audio_new_s:.4f}  "
                  f"(自报推理 {pure:.1f}s / 新增音频 {audio_new_s:.1f}s)")
            pure_f.write_text(f"{pure / audio_new_s:.4f}\n")
        elif pure is None:
            print(f"    [note] {args.model} 无自报推理计时，纯 RTF 不可得（仅 e2e）")


if __name__ == "__main__":
    main()
