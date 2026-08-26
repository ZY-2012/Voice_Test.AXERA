#!/usr/bin/env python3
"""
将 aria2c 下载的 parquet 测试集"解压"成 wav 音频 + 文本清单
用法: python extract_parquet.py [--base /data/shared/huyuan/8860_datasets]
输出: <base>/<subset>/wav/*.wav 和 <base>/<subset>/transcripts.txt (id<TAB>text)
"""
import argparse
import io
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


def extract_voxpopuli(parquet_path: Path, out_dir: Path):
    """voxpopuli: 音频在 audio 字段（datasets 自动解码），文本用 normalized_text"""
    ds = load_dataset("parquet", data_files=str(parquet_path))
    print(f"voxpopuli: {len(ds)} 条样本")
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transcripts.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            audio = item["audio"]
            wav_path = out_dir / "wav" / f"{item['audio_id']}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
            f.write(f"{item['audio_id']}\t{item['normalized_text']}\n")
            if (i + 1) % 500 == 0:
                print(f"  进度 {i+1}/{len(ds)}")
    print(f"  -> {out_dir / 'wav'} + transcripts.txt")


def extract_tedlium(parquet_path: Path, out_dir: Path):
    """tedlium: 音频在 context 字段（字段名不含 audio，datasets 不自动解码，手动处理 bytes），文本在 answer"""
    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path)).to_pandas()
    print(f"tedlium: {len(table)} 条样本")
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transcripts.txt", "w", encoding="utf-8") as f:
        for i, row in table.iterrows():
            wav_path = out_dir / "wav" / f"tedlium_{i:05d}.wav"
            sf.write(str(wav_path), sf.read(io.BytesIO(row["context"]["bytes"]))[0],
                     16000)  # TED-LIUM 3 为 16kHz
            f.write(f"tedlium_{i:05d}\t{row['answer']}\n")
            if (i + 1) % 500 == 0:
                print(f"  进度 {i+1}/{len(table)}")
    print(f"  -> {out_dir / 'wav'} + transcripts.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/shared/huyuan/8860_datasets")
    args = parser.parse_args()
    base = Path(args.base)

    vp = base / "voxpopuli" / "test-00000-of-00001.parquet"
    if vp.exists():
        extract_voxpopuli(vp, base / "voxpopuli")
    else:
        print(f"跳过 voxpopuli（未找到 {vp}）")

    td = base / "tedlium" / "test-00000-of-00001.parquet"
    if td.exists():
        extract_tedlium(td, base / "tedlium")
    else:
        print(f"跳过 tedlium（未找到 {td}）")


if __name__ == "__main__":
    main()
