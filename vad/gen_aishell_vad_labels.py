#!/usr/bin/env python3
"""
生成中文 VAD 测试集（AISHELL-1）的采样级标签:
  用 funasr fsmn-vad 模型 (iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
  对每条语音做端点检测, 输出语音段 [start_ms, end_ms],
  再转换为 16kHz 采样级二值标签: 1=语音 0=静音 (与 LibriVAD 英文标签同约定)

注意: 与英文 LibriVAD 的强制对齐标签不同, 中文标签为 VAD 模型生成 (无人工对齐可用),
      在文档中明确标注为 "模型生成标签"。

输出 (vad/aishell1/test/):
  labels/<utt_id>.npy    采样级 0/1 标签 (int16, 长度=音频长度)
  test.scp               每行: utt_id<TAB>wav路径<TAB>label路径
用法: python gen_aishell_vad_labels.py [--split test|dev] [--device cuda:0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config
ROOT = Path(load_config()["paths"]["data_root"])
ASR_WAV_DIR = ROOT / "asr" / "aishell1" / "wav"          # 已整理的 test (7176 条)
AISHELL_RAW = ROOT / "vad" / "aishell1" / "speech_asr_aishell1_testsets"
OUT_DIR = ROOT / "vad" / "aishell1"

MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from funasr import AutoModel
    print(f"加载模型 {MODEL} ({args.device}) ...")
    model = AutoModel(model=MODEL, device=args.device, disable_update=True)

    if args.split == "test":
        wavs = sorted(ASR_WAV_DIR.glob("*.wav"))
    else:
        wavs = sorted((AISHELL_RAW / "wav" / "dev").rglob("*.wav"))
    print(f"{args.split} 音频: {len(wavs)} 条")

    out = OUT_DIR / args.split
    (out / "labels").mkdir(parents=True, exist_ok=True)
    scp_lines = []
    n_seg_total = 0
    for i, wav_path in enumerate(wavs):
        utt_id = wav_path.stem
        res = model.generate(input=str(wav_path))
        label = np.zeros(sf.info(wav_path).frames, dtype=np.int16)
        if res and "value" in res[0]:
            for seg in res[0]["value"]:
                s_ms, e_ms = seg[0], seg[1]
                s_s = int(s_ms / 1000 * 16000)
                e_s = int(e_ms / 1000 * 16000)
                label[s_s:min(e_s, len(label))] = 1
                n_seg_total += 1
        dst = out / "labels" / f"{utt_id}.npy"
        np.save(dst, label)
        scp_lines.append(f"{utt_id}\t{wav_path}\t{dst}")
        if (i + 1) % 500 == 0:
            print(f"  进度 {i+1}/{len(wavs)}, 平均语音段/条: {n_seg_total/(i+1):.2f}")

    (out / "test.scp").write_text("\n".join(scp_lines) + "\n", encoding="utf-8")
    n_speech = sum(1 for _ in scp_lines if np.load(_.split("\t")[2]).any())
    print(f"完成: {len(scp_lines)} 条 -> {out}")
    print(f"含语音的样本: {n_speech} 条, 平均语音段/条: {n_seg_total/len(scp_lines):.2f}")


if __name__ == "__main__":
    main()
