#!/usr/bin/env python3
"""MeloTTS 载入一次批量合成（板端）：语言模块/encoder/decoder 只加载一次，
逐条合成 benchmark TTS 子集文本，热启动 RTF（warmup 1 条后只计推理）。
用法: python melotts_batch.py --dataset aishell3 [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config

MODEL_DIR = Path(load_config()["paths"]["model_root"]) / "MeloTTS" / "python"
sys.path.insert(0, str(MODEL_DIR))
import melotts as M


def synthesize_once(sess_enc, sess_dec, g, symbol_to_id, language, speed, dec_len, text):
    sens = M.split_sentences_into_pieces(text, language, quiet=True)
    audio_list = []
    for se in sens:
        if language in ["EN", "ZH_MIX_EN"]:
            se = M.re.sub(r"([a-z])([A-Z])", r"\1 \2", se)
        phones, tones, lang_ids, norm_text, word2ph = M.get_text_for_tts_infer(
            se, language, symbol_to_id=symbol_to_id)
        z_p, pronoun_lens, audio_len = sess_enc.run(None, input_feed={
            "phone": phones, "g": g, "tone": tones, "language": lang_ids,
            "noise_scale": np.array([0], dtype=np.float32),
            "length_scale": np.array([1.0 / speed], dtype=np.float32),
            "noise_scale_w": np.array([0], dtype=np.float32),
            "sdp_ratio": np.array([0], dtype=np.float32)})
        word2pronoun = M.calc_word2pronoun(word2ph, pronoun_lens)
        pn_slices, zp_slices = M.generate_slices(word2pronoun, dec_len)
        audio_len = audio_len[0]
        sub_audio_list = []
        for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
            zp_slice = z_p[..., zs]
            sub_dec_len = zp_slice.shape[-1]
            sub_audio_len = 512 * sub_dec_len
            if zp_slice.shape[-1] < dec_len:
                zp_slice = np.concatenate(
                    (zp_slice, np.zeros((*zp_slice.shape[:-1], dec_len - zp_slice.shape[-1]),
                                        dtype=np.float32)), axis=-1)
            audio = sess_dec.run(None, input_feed={"z_p": zp_slice, "g": g})[0].flatten()
            audio_start = 0
            if len(sub_audio_list) > 0 and pn_slices[i - 1].stop > ps.start:
                audio_start = 512 * word2pronoun[ps.start]
            audio_end = sub_audio_len
            if i < len(pn_slices) - 1 and ps.stop > pn_slices[i + 1].start:
                audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]
            sub_audio_list.append(audio[audio_start:audio_end])
        audio_list.append(M.merge_sub_audio(sub_audio_list, 0, audio_len))
    return M.audio_numpy_concat(audio_list, sr=44100, speed=speed)


def main():
    import onnxruntime as ort
    import axengine as axe
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["aishell3", "ljspeech"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.8)
    args = ap.parse_args()
    lang = "zh" if args.dataset == "aishell3" else "en"
    language = "ZH_MIX_EN" if lang == "zh" else "EN"   # melotts.py 默认中文语言

    text_f = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / args.dataset / "text"
    rows = [l.split("\t", 1) for l in text_f.read_text().splitlines() if "\t" in l]
    if args.limit > 0:
        rows = rows[:args.limit]
    out_dir = Path(cfg["paths"]["data_root"]) / "benchmark" / "tts" / "syn" / f"melotts_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "zh" if lang == "zh" else "en"           # encoder/decoder 文件名后缀
    g_file = f"g-{language.lower()}.bin"               # g 文件名按语言（如 g-zh_mix_en.bin）
    sess_enc = ort.InferenceSession(str(MODEL_DIR.parent / "encoder-onnx" / f"encoder-{suffix}.onnx"),
                                    providers=["CPUExecutionProvider"])
    sess_dec = axe.InferenceSession(str(MODEL_DIR.parent / "decoder-ax650" / f"decoder-{suffix}.axmodel"))
    g = np.fromfile(str(MODEL_DIR.parent / g_file), dtype=np.float32).reshape(1, 256, 1)
    symbol_to_id = {s: i for i, s in enumerate(M.LANG_TO_SYMBOL_MAP[language])}
    dec_len = 128
    print(f"MeloTTS {language}: 模型载入完成, {len(rows)} 条待合成")

    def run_one(text):
        return synthesize_once(sess_enc, sess_dec, g, symbol_to_id, language,
                               args.speed, dec_len, text)

    infer_s = audio_s = 0.0
    ok = 0
    for k, (uid, text) in enumerate(rows):
        outp = out_dir / f"{uid}.wav"
        t0 = time.perf_counter()
        wav = run_one(text)
        dt = time.perf_counter() - t0
        if k > 0:  # warmup 跳过首条
            infer_s += dt
            audio_s += len(wav) / 44100
        sf.write(outp, wav, 44100)
        ok += 1
        if (k + 1) % 20 == 0:
            print(f"  进度 {k+1}/{len(rows)}")
    rtf = infer_s / audio_s if audio_s else float("nan")
    print(f">>> melotts/{args.dataset}: 合成 {ok}/{len(rows)} 条 -> {out_dir}  热启动RTF={rtf:.4f}")
    (out_dir / "_rtf.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
