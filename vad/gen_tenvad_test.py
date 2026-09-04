#!/usr/bin/env python3
"""
生成 TEN VAD 官方测试集标签（30 条 16kHz 音频 + 段级标注）:
  源数据随 TEN VAD 模型仓库发布（AXERA-TECH/ten-vad 的 testset/），
  每条 .scv 为单行 CSV: utt_id,start,end,label,start,end,label,...
  （秒为单位的段边界；label 1=语音 0=静音）
  转换为采样级(16kHz)二值标签 .npy，与 librivad/aishell1 同约定，
  这样 eval_silero.py / eval_tenvad.py 无需区分数据集。

输入（按优先级找）:
  <model_root>/ten-vad/testset/          # 模型仓库自带（download_models.sh 已拉）
  <data_root>/vad/tenvad/raw/testset/    # 或 download_datasets.sh 单独下的副本

输出 (<data_root>/vad/tenvad/test/):
  wav/<utt_id>.wav        16k/mono/PCM16 音频（从源拷贝）
  labels/<utt_id>.npy     采样级 0/1 标签
  test.scp                每行: utt_id<TAB>wav路径<TAB>label路径
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config

_CFG = load_config()
DATA_ROOT = Path(_CFG["paths"]["data_root"])
MODEL_ROOT = Path(_CFG["paths"]["model_root"])
OUT_DIR = DATA_ROOT / "vad" / "ten_official" / "test"
SR = 16000


def find_src():
    """定位官方 testset 目录（模型仓库自带优先）。"""
    for cand in (MODEL_ROOT / "ten-vad" / "testset",
                 DATA_ROOT / "vad" / "ten_official" / "raw" / "testset",
                 DATA_ROOT / "vad" / "ten_official" / "raw"):
        if cand.is_dir() and any(cand.glob("*.scv")):
            return cand
    return None


def scv_to_samples(scv_path, n_samples, sr=SR):
    """单行 scv（id + (start,end,label) 三元组）→ 采样级 0/1 标签。"""
    fields = scv_path.read_text(encoding="utf-8").strip().split(",")
    body = fields[1:]                      # 去掉首列 utt_id
    label = np.zeros(n_samples, dtype=np.int16)
    for i in range(0, len(body) - 2, 3):
        start, end, flag = float(body[i]), float(body[i + 1]), int(float(body[i + 2]))
        if flag != 1:                      # 只有 1 是语音
            continue
        s, e = int(start * sr), int(end * sr)
        label[max(0, s):min(n_samples, e)] = 1
    return label


def main():
    src = find_src()
    if src is None:
        print("  跳过 TEN VAD 测试集：未找到 testset/（先跑 download_models.sh vad）")
        return
    (OUT_DIR / "wav").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    scp_lines, skipped = [], 0
    for wav_path in sorted(src.glob("*.wav")):
        utt_id = wav_path.stem
        scv = src / f"{utt_id}.scv"
        if not scv.is_file():
            skipped += 1
            continue
        info = sf.info(wav_path)
        if info.samplerate != SR or info.channels != 1:
            print(f"  [warn] {utt_id}: {info.samplerate}Hz/{info.channels}ch 非 16k 单声道，跳过")
            skipped += 1
            continue
        dst_wav = OUT_DIR / "wav" / f"{utt_id}.wav"
        if not dst_wav.exists():
            shutil.copy(wav_path, dst_wav)
        dst_lab = OUT_DIR / "labels" / f"{utt_id}.npy"
        np.save(dst_lab, scv_to_samples(scv, info.frames))
        scp_lines.append(f"{utt_id}\t{dst_wav}\t{dst_lab}")

    (OUT_DIR / "test.scp").write_text("\n".join(scp_lines) + "\n", encoding="utf-8")
    print(f"TEN VAD 测试集: {len(scp_lines)} 条（跳过 {skipped}） -> {OUT_DIR}")


if __name__ == "__main__":
    main()
