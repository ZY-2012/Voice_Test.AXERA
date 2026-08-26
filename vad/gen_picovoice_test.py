#!/usr/bin/env python3
"""
生成 Picovoice voice-activity-benchmark 测试集（与官方完全一致的逻辑）:
  - 语音: LibriSpeech test-clean（复用 asr 目录已提取的 16k wav，按官方排序）
  - 噪声: DEMAND（排除官方 blocklist 的 8 个含人声环境），0dB SNR
  - 拼接: 每个 utterance 之间 20 秒静音
  - 标签: 512 采样点(32ms)/帧, 三值 (0=静音 1=未知 2=语音), 能量检测+±5帧边界平滑
  - 随机种子 778（与官方一致，噪声选择可复现）

输出:
  vad/picovoice/test_audio.wav       混合后的长音频 (约 5.4h)
  vad/picovoice/benchmark_labels.txt 每帧一行标签 (约 61 万行)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # dataset.py / mixer.py（官方，随仓库）
sys.path.insert(0, str(_HERE.parent / "tools"))
from common import load_config  # noqa: E402
_DR = Path(load_config()["paths"]["data_root"])
VAD_DIR = _DR / "vad"
ASR_WAV_DIR = _DR / "asr" / "librispeech" / "wav"
DEMAND_DIR = VAD_DIR / "demand" / "extracted" / "demand"
OUT_DIR = VAD_DIR / "picovoice"

from dataset import DEMANDDataset  # noqa: E402  (官方 dataset.py)
from mixer import create_test_files  # noqa: E402  (官方 mixer.py)


class LibriSpeechWavDataset:
    """等价于官方 LibriSpeechDataset: 按路径排序的 test-clean 音频列表"""

    def __init__(self, wav_dir: Path, test_clean_ids):
        paths = [wav_dir / f"{utt}.wav" for utt in test_clean_ids]
        self._paths = sorted(str(p) for p in paths if p.exists())
        assert len(self._paths) == 2620, f"test-clean 应有 2620 条, 实际 {len(self._paths)}"
        self._random = np.random.RandomState(seed=778)

    def get(self, index, dtype=np.float32):
        pcm, sr = soundfile.read(self._paths[index], dtype=dtype)
        assert sr == 16000
        return pcm

    def random(self, dtype=np.float32):
        return self.get(self._random.randint(0, self.size()), dtype)

    def size(self):
        return len(self._paths)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # test-clean 的权威列表来自 LibriVAD 强制对齐文件 (2620 条)
    align_dir = VAD_DIR / "librivad" / "alignments" / "librispeech_alignments" / "test-clean"
    test_clean_ids = sorted(tg.stem for tg in align_dir.rglob("*.TextGrid"))
    speech_dataset = LibriSpeechWavDataset(ASR_WAV_DIR, test_clean_ids)
    noise_dataset = DEMANDDataset(str(DEMAND_DIR))
    print(f"语音样本: {speech_dataset.size()} 条")
    print(f"噪声文件: {noise_dataset.size()} 个 (已排除含人声环境)")

    speech_path = str(OUT_DIR / "test_audio.wav")
    label_path = str(OUT_DIR / "benchmark_labels.txt")
    create_test_files(speech_path, label_path, speech_dataset, noise_dataset, snr_db=0)

    import soundfile as sf
    info = sf.info(speech_path)
    n_frames = sum(1 for _ in open(label_path))
    print(f"输出: {speech_path}")
    print(f"  时长: {info.duration/3600:.2f} 小时 @ {info.samplerate}Hz")
    print(f"  标签帧数: {n_frames} (每帧 512 采样点 = 32ms)")


if __name__ == "__main__":
    main()
