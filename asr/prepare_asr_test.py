#!/usr/bin/env python3
"""
整理 ESB 测试数据为 ASR 可测试格式:
  每个子集输出:
    <subset>/wav/<utt_id>.wav   16kHz 单声道 PCM wav
    <subset>/wav.scp            utt_id \t 绝对路径
    <subset>/text               utt_id \t 转录文本
(Kaldi/funasr 风格, 可直接用于 ASR 模型测试)

自动跳过未下载完的文件(存在 .aria2 即视为下载中)。
"""
import argparse
import csv
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

WAV_DIR = "wav"

# ============ 文本清洗 ============
GIGASPEECH_PUNCT = {" <comma>": ",", " <period>": ".", " <questionmark>": "?",
                    " <exclamationpoint>": "!", " <semicolon>": ";", " <colon>": ":"}


def replace_case_insensitive(text, mapping):
    for src, dst in mapping.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
    return text
GIGASPEECH_JUNK = ["<other>", "<sil>"]
EARNINGS_JUNK = ["<noise>", "<crosstalk>", "<affirmative>", "<inaudible>", "inaudible",
                 "<laugh>", "<silence>"]
# ESB 官方脚本: 整段文本仅含噪声标记的段直接跳过
IGNORE_SEGMENTS = {"ignore_time_segment_in_scoring", "<noise>", "<music>", "[noise]",
                   "[laughter]", "[silence]", "[vocalized-noise]", "<crosstalk>",
                   "<affirmative>", "<inaudible>", "<laugh>", "", "<other>", "<sil>"}


def clean_text(text, extra_junk=()):
    for tok in extra_junk:
        text = text.replace(tok, "")
    text = re.sub(r"\s\s+", " ", text).strip()
    return text


def to_16k(samples, sr):
    """必要时重采样到 16kHz, 返回 int16 mono"""
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        try:
            import torch
            import torchaudio.functional as F
            t = torch.from_numpy(samples.astype(np.float32) / 32768.0).unsqueeze(0)
            samples = (F.resample(t, sr, 16000).squeeze(0).numpy() * 32768.0).astype(np.int16)
        except Exception as e:
            print(f"    [重采样失败 {sr}->16000: {e}], 按原采样率保存")
    else:
        if np.issubdtype(samples.dtype, np.floating):
            # float 音频 (如 datasets 解码结果) 通常在 [-1,1], 需缩放到 int16
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).astype(np.int16)
        else:
            samples = samples.astype(np.int16)
    return samples, 16000


def emit(utt_id, samples, sr, text, out_dir, scp_lines, text_lines):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", utt_id)
    wav_path = out_dir / f"{safe_id}.wav"
    sf.write(str(wav_path), samples, sr)
    scp_lines.append(f"{safe_id}\t{wav_path}")
    text_lines.append(f"{safe_id}\t{text}")


def read_audio(path):
    # 统一按 float 读（[-1,1] 范围）: 有些源是 32bit float wav（如 AMI），
    # 强制 dtype='int16' 读取会得到全零
    samples, sr = sf.read(str(path))
    return to_16k(samples, sr)


def write_manifest(subset_dir, scp_lines, text_lines):
    (subset_dir / "wav.scp").write_text("\n".join(scp_lines) + "\n", encoding="utf-8")
    (subset_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"  总计 {len(text_lines)} 条 -> {subset_dir / 'wav.scp'} + {subset_dir / 'text'}")


# ============ 各子集处理 ============

def prep_librispeech(sub):
    print(">>> librispeech")
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    for tar_name in ["test-clean.tar.gz", "test-other.tar.gz"]:
        tar_path = sub / tar_name
        if not tar_path.exists():
            print(f"   跳过 {tar_name} (不存在)"); continue
        tmp = sub / ("_extract_" + tar_name.replace(".tar.gz", ""))
        tmp.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        for trans_file in sorted(tmp.rglob("*.trans.txt")):
            texts = {}
            for line in trans_file.read_text(encoding="utf-8").splitlines():
                if " " in line:
                    uid, t = line.split(" ", 1)
                    texts[uid] = t
            for flac in sorted(trans_file.parent.glob("*.flac")):
                uid = flac.stem
                if uid not in texts:
                    continue
                samples, sr = read_audio(flac)
                emit(uid, samples, sr, texts[uid], out, scp, txt)
        import shutil; shutil.rmtree(tmp)
    write_manifest(sub, scp, txt)


def prep_voxpopuli(sub):
    print(">>> voxpopuli")
    from datasets import load_dataset
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    ds = load_dataset("parquet", data_files=str(sub / "test-00000-of-00001.parquet"))
    ds = ds["train"] if isinstance(ds, dict) else ds
    for item in ds:
        audio = item["audio"]
        samples, sr = to_16k(np.asarray(audio["array"]), audio["sampling_rate"])
        emit(item["audio_id"], samples, sr, item["normalized_text"], out, scp, txt)
    write_manifest(sub, scp, txt)


def prep_tedlium(sub):
    print(">>> tedlium")
    from datasets import load_dataset
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    ds = load_dataset("parquet", data_files=str(sub / "test-00000-of-00001.parquet"))
    ds = ds["train"] if isinstance(ds, dict) else ds
    for i, item in enumerate(ds):
        audio = item.get("context")
        if audio and "array" in audio:
            samples, sr = to_16k(np.asarray(audio["array"]), audio["sampling_rate"])
        else:  # 兜底: 手动解码 bytes
            import pyarrow.parquet as pq
            table = pq.read_table(str(sub / "test-00000-of-00001.parquet")).to_pandas()
            samples, sr = sf.read(io.BytesIO(table.iloc[i]["context"]["bytes"]), dtype="int16")
            samples, sr = to_16k(samples, sr)
        emit(f"tedlium_{i:05d}", samples, sr, item["answer"], out, scp, txt)
    write_manifest(sub, scp, txt)


def prep_earnings22(sub):
    print(">>> earnings22")
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    meta = {}
    with open(sub / "metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["file"]] = row["sentence"]
    for tar_path in sorted((sub / "chunked").glob("*.tar.gz")):
        if Path(str(tar_path) + ".aria2").exists():
            print(f"   跳过 {tar_path.name} (下载中)"); continue
        chunk = tar_path.name.split(".")[0]  # 4432298.tar.gz -> 4432298
        with tarfile.open(tar_path) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".wav"):
                    continue
                rel = m.name.lstrip("./")  # e.g. 4432298/0.wav
                text = meta.get(rel)
                if text is None:
                    continue
                text = clean_text(text, EARNINGS_JUNK)
                if not text:
                    continue
                f = tf.extractfile(m)
                samples, sr = sf.read(io.BytesIO(f.read()), dtype="int16")
                samples, sr = to_16k(samples, sr)
                seg = Path(rel).stem
                emit(f"{chunk}_{seg}", samples, sr, text, out, scp, txt)
    write_manifest(sub, scp, txt)


def prep_ami(sub):
    print(">>> ami")
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    ann = {}
    with open(sub / "annotations" / "text", encoding="utf-8") as f:
        for line in f:
            if " " in line:
                uid, t = line.split(" ", 1)
                ann[uid] = t.strip()
    print(f"  标注段数: {len(ann)}")
    wav_files = {}
    for tar_path in sorted((sub / "audio").glob("*.tar.gz")):
        if Path(str(tar_path) + ".aria2").exists():
            print(f"   跳过 {tar_path.name} (下载中)"); continue
        tmp = sub / "_extract_tmp"
        tmp.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        for wav in tmp.rglob("*.wav"):
            wav_files[wav.name] = wav
    print(f"  音频文件数: {len(wav_files)}")
    missing = 0
    for uid, text in ann.items():
        # AMI_EN2002a_H00_MEE073_0000096_0000665 -> eval_ami_en2002a_h00_mee073_0000096_0000665.wav
        wav_name = "eval_" + uid.lower() + ".wav"
        src = wav_files.get(wav_name)
        if src is None:
            missing += 1
            continue
        samples, sr = read_audio(src)
        emit(uid, samples, sr, text, out, scp, txt)
    import shutil; shutil.rmtree(sub / "_extract_tmp", ignore_errors=True)
    if missing:
        print(f"  [注意] {missing} 条标注无对应音频文件")
    write_manifest(sub, scp, txt)


def prep_gigaspeech(sub):
    print(">>> gigaspeech")
    out = sub / WAV_DIR; out.mkdir(parents=True, exist_ok=True)
    scp, txt = [], []
    for tar_path in sorted((sub / "audio").glob("test_chunks_*.tar.gz")):
        if Path(str(tar_path) + ".aria2").exists():
            print(f"   跳过 {tar_path.name} (下载中)"); continue
        chunk_no = re.search(r"(\d{4})", tar_path.name).group(1)
        meta_csv = sub / "meta" / f"test_chunks_{chunk_no}_metadata.csv"
        if not meta_csv.exists():
            print(f"   跳过 {tar_path.name} (无对应 metadata)"); continue
        texts = {}
        with open(meta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                texts[row["sid"]] = row["text_tn"]
        tmp = sub / f"_extract_{chunk_no}"
        tmp.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        for wav in sorted(tmp.rglob("*.wav")):
            sid = wav.stem
            text = texts.get(sid)
            if text is None:
                continue
            text = text.lower()  # 与 ESB 官方预处理一致: gigaspeech 文本统一小写
            if text in IGNORE_SEGMENTS:  # 纯噪声段跳过 (ESB 同样处理)
                continue
            for punc_from, punc_to in GIGASPEECH_PUNCT.items():
                text = replace_case_insensitive(text, {punc_from: punc_to})
            for junk in GIGASPEECH_JUNK:
                text = re.sub(re.escape(junk), "", text, flags=re.IGNORECASE)
            text = clean_text(text)
            if not text:
                continue
            samples, sr = read_audio(wav)
            emit(sid, samples, sr, text, out, scp, txt)
        import shutil; shutil.rmtree(tmp)
    write_manifest(sub, scp, txt)


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/data/shared/huyuan/8860_datasets/asr")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="只处理指定子集, 如: librispeech voxpopuli tedlium earnings22 ami gigaspeech")
    args = parser.parse_args()
    base = Path(args.base)

    prep = {
        "librispeech": prep_librispeech,
        "voxpopuli": prep_voxpopuli,
        "tedlium": prep_tedlium,
        "earnings22": prep_earnings22,
        "ami": prep_ami,
        "gigaspeech": prep_gigaspeech,
    }
    for name, fn in prep.items():
        if args.subsets and name not in args.subsets:
            continue
        sub = base / name
        if not sub.exists():
            print(f"跳过 {name} (目录不存在)"); continue
        fn(sub)

    print("\n>>> 整理完成。各子集下: wav/ 音频 + wav.scp + text")


if __name__ == "__main__":
    main()
