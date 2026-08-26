#!/usr/bin/env python3
"""
生成 LibriVAD test 标签集（与官方 create_labels.py 完全一致的逻辑）:
  - 对每条 test-clean 音频, 解析其强制对齐 TextGrid (词级时间戳)
  - 生成采样级(16kHz)二值标签: 1=语音 0=静音, 存为 .npy
  - 标签长度 = 最后一个词结束时间 × 16000 (官方行为, 比 wav 略短, 尾部静音不计)

输出 (vad/librivad/test/):
  wav/<utt_id>.wav           从 asr 目录复制的 16k wav
  labels/<utt_id>.npy        采样级 0/1 标签
  test.scp                   每行: utt_id<TAB>wav路径<TAB>label路径
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config
_DR = Path(load_config()["paths"]["data_root"])
VAD_DIR = _DR / "vad"
ASR_WAV_DIR = _DR / "asr" / "librispeech" / "wav"
ALIGN_ROOT = VAD_DIR / "librivad" / "alignments" / "librispeech_alignments"
OUT_DIR = VAD_DIR / "librivad" / "test"


def generate_label_from_timeframes(timeframes, words, sample_rate=16000):
    """官方 create_labels.py::_generate_label_from_timeframes 原样实现"""
    if timeframes.size == 0:
        return np.array([], dtype=np.int16)

    total_samples = int(timeframes[-1] * sample_rate)
    label = np.ones(total_samples, dtype=np.int16)

    is_silent = [w == '' or w is None for w in words]
    silent_indices = np.where(is_silent)[0]

    if silent_indices.size > 0:
        if silent_indices[0] == 0:
            end_sample = int(timeframes[0] * sample_rate)
            label[0:end_sample] = 0
        for j in silent_indices:
            if j > 0:
                start_sample = int(timeframes[j - 1] * sample_rate)
                end_sample = int(timeframes[j] * sample_rate)
                label[start_sample:end_sample] = 0
    return label


def main():
    import textgrids

    (OUT_DIR / "wav").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    wavs = sorted(ASR_WAV_DIR.glob("*.wav"))
    print(f"test-clean wav: {len(wavs)} 条")

    done = skipped = 0
    scp_lines = []
    for wav_path in wavs:
        utt_id = wav_path.stem  # 如 1089-134686-0000
        spk, chap, _ = utt_id.split("-")
        tg = ALIGN_ROOT / "test-clean" / spk / chap / f"{utt_id}.TextGrid"
        if not tg.is_file():
            skipped += 1
            continue

        grid = textgrids.TextGrid(tg)
        words_tier = grid["words"]
        timeframes = np.array([iv.xmax for iv in words_tier], dtype=np.float32)
        words = [iv.text for iv in words_tier]

        label = generate_label_from_timeframes(timeframes, words)

        dst_wav = OUT_DIR / "wav" / f"{utt_id}.wav"
        if not dst_wav.exists():
            shutil.copy(wav_path, dst_wav)
        dst_lab = OUT_DIR / "labels" / f"{utt_id}.npy"
        np.save(dst_lab, label)

        scp_lines.append(f"{utt_id}\t{dst_wav}\t{dst_lab}")
        done += 1

    (OUT_DIR / "test.scp").write_text("\n".join(scp_lines) + "\n", encoding="utf-8")
    print(f"生成标签: {done} 条, 无对齐文件跳过: {skipped} 条")
    print(f"输出: {OUT_DIR}/wav + {OUT_DIR}/labels + {OUT_DIR}/test.scp")


if __name__ == "__main__":
    main()
