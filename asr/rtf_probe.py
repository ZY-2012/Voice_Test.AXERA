#!/usr/bin/env python3
"""ASR python 推理-only RTF 探测（单进程：模型载入一次 + warmup 1 条 + 计时 9 条）。
不含模型加载/初始化；与 C++ CLI 的 RTF 口径对齐（C++ 亦不含 load）。
用法: python rtf_probe.py whisper-tiny|whisper-base|whisper-small|whisper-turbo|sensevoice|firered
"""
import sys
import time
from pathlib import Path

import soundfile as sf

EXPORT = Path("/root/huyuan/workspace/8860_export")
BENCH = Path("/root/huyuan/workspace/8860_datasets/benchmark/asr/aishell1")
GTP = (BENCH / "ground_truth.txt").read_text().splitlines()
WAVS = [BENCH / "aishell_S0764" / (l.split()[0] + ".wav") for l in GTP][:10]
DURS = [sf.info(str(w)).duration for w in WAVS]


def time_it(fn, *a):
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


def run_whisper(variant):
    sys.path.insert(0, str(EXPORT / "Whisper" / "python"))
    from whisper_ax import Whisper
    model = Whisper(variant, str(EXPORT / "Whisper" / "models-ax650"), "zh", "transcribe")
    time_it(model.run, str(WAVS[0]))          # warmup
    s = sum(time_it(model.run, str(w)) for w in WAVS[1:])
    return s / sum(DURS[1:])


def run_sensevoice():
    sys.path.insert(0, str(EXPORT / "SenseVoice" / "python"))
    from SenseVoiceAx import SenseVoiceAx
    root = EXPORT / "SenseVoice" / "python" / "models" / "SenseVoice" / "sensevoice_ax650"
    model = SenseVoiceAx(str(root / "sensevoice.axmodel"), str(root / "am.mvn"),
                         str(root / "tokens.txt"), max_seq_len=256, beam_size=3)
    time_it(model.infer, str(WAVS[0]), "zh", False)   # warmup
    s = sum(time_it(model.infer, str(w), "zh", False) for w in WAVS[1:])
    return s / sum(DURS[1:])


def run_firered():
    sys.path.insert(0, str(EXPORT / "FireRedASR-AED"))
    from fireredasr_axmodel import FireRedASRAxModel
    ax = EXPORT / "FireRedASR-AED" / "axmodel"
    model = FireRedASRAxModel(str(ax / "encoder.axmodel"), str(ax / "decoder_loop.axmodel"),
                              str(ax / "cmvn.ark"), str(ax / "dict.txt"),
                              str(ax / "train_bpe1000.model"), decode_max_len=128, audio_dur=10)
    time_it(model.transcribe, [str(WAVS[0])], 1, 1)   # warmup
    s = sum(time_it(model.transcribe, [str(w)], 1, 1) for w in WAVS[1:])
    return s / sum(DURS[1:])


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind.startswith("whisper-"):
        rtf = run_whisper(kind.split("-")[1])
    elif kind == "sensevoice":
        rtf = run_sensevoice()
    elif kind == "firered":
        rtf = run_firered()
    else:
        raise SystemExit("unknown model")
    print(f"{kind} python_RTF={rtf:.4f}  (9条, 载入一次, 不含加载)")
