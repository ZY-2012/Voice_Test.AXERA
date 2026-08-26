#!/usr/bin/env python3
"""SE 数据制作：VoiceBank-DEMAND test parquet → clean/ + noisy/ 16k wav + test.scp
用法: python prepare_se.py [--data_root ...]
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import load_config  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=cfg["paths"]["data_root"])
    args = ap.parse_args()
    base = Path(args.data_root) / "se" / "voicebank_demand"
    pq = base / "test-00000-of-00001.parquet"
    if not pq.exists():
        print(f"跳过 SE：未找到 {pq}（先跑 download_datasets.sh se）")
        return

    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pqt

    (base / "clean").mkdir(exist_ok=True)
    (base / "noisy").mkdir(exist_ok=True)
    t = pqt.read_table(pq).to_pandas()
    lines = []
    for _, row in t.iterrows():
        uid = row["id"]
        c, csr = sf.read(io.BytesIO(row["clean"]["bytes"]))
        n, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]))
        if c.ndim > 1:
            c = c.mean(1)
        if n.ndim > 1:
            n = n.mean(1)
        cf, nf = base / "clean" / f"{uid}.wav", base / "noisy" / f"{uid}.wav"
        sf.write(cf, c, csr)
        sf.write(nf, n, csr)
        lines.append(f"{uid}\t{cf}\t{nf}")
    (base / "test.scp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SE] VoiceBank-DEMAND: {len(lines)} 对 clean/noisy -> {base}")


if __name__ == "__main__":
    main()
