#!/usr/bin/env python3
"""MeloTTS 载入一次批量合成（板端）：语言模块/encoder/decoder 只加载一次，
逐条合成 benchmark TTS 子集文本，热启动 RTF（warmup 1 条后只计推理）。
用法: python melotts_batch.py --dataset aishell3 [--limit N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from common import load_config, TTS_DATASETS, tts_lang

MODEL_DIR = Path(load_config()["paths"]["model_root"]) / "MeloTTS" / "python"
sys.path.insert(0, str(MODEL_DIR))

# 英文经 g2p_en 做词性标注，需 NLTK 数据（averaged_perceptron_tagger_eng + cmudict）。
# 板端直连 nltk 服务器常挂住并留下 *.zip.lock，故把数据放在 model_root 下随模型一起分发，
# 通过 NLTK_DATA 指过去（NLTK 只搜固定的几个本地目录 + 该环境变量）。
# 准备一次即可：host 侧 python -c "import nltk;
#   nltk.download('averaged_perceptron_tagger_eng', download_dir='<model_root>/_nltk_data');
#   nltk.download('cmudict', download_dir='<model_root>/_nltk_data')"
_NLTK_DIR = MODEL_DIR.parent.parent / "_nltk_data"
if _NLTK_DIR.is_dir():
    os.environ["NLTK_DATA"] = (f"{_NLTK_DIR}{os.pathsep}{os.environ['NLTK_DATA']}"
                               if os.environ.get("NLTK_DATA") else str(_NLTK_DIR))

# vendor melotts 的 text/ 用**相对路径**加载 BERT（model_id='bert-base-multilingual-uncased'
# 实为 MeloTTS/python/ 下的同名目录），故必须先 chdir 到 MODEL_DIR 再 import，
# 否则报找不到模型。synth_batch 走子进程 cwd=MODEL_DIR 所以不受影响。
# 本脚本所有输入/输出路径均来自 config（绝对路径），chdir 安全。
os.chdir(MODEL_DIR)
import melotts as M

# 单段合成音时长上限（秒）。实测 melotts 对超长单段会**静默截断**，天花板约 13.58s：
# LibriSpeech 200 条里 35~45 词与 45~80 词两桶的合成音均长都停在 ~12.8s / 最长 13.5s，
# 时长不再随文本增长；其中 51 条出现句尾内容丢失（删除率 11.6%，时长比 0.85），
# 而 149 条短句删除率为 0、时长比 1.20 完全正常。
# 根因链：vendor `split_sentences_latin` 走 `txtsplit(text, 256, 512)` 按**字符数**切分，
# 而 LibriSpeech 文本无标点且常在 256 字符以内 ⇒ 整句不切分 ⇒ 单段超限被截。
# 中文因 `split_sentences_zh` 按标点+min_len 细切、且句子短，从不触发。
# 故此处按 encoder **自己预测的** audio_len 再做一次保险切分（见 _fit_pieces）。
MAX_SEG_SEC = 12.0


def _encode(sess_enc, g, symbol_to_id, language, speed, text):
    """文本 -> encoder 输出。audio_len 是模型预测的总时长（采样点数）。"""
    if language in ["EN", "ZH_MIX_EN"]:
        text = M.re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    phones, tones, lang_ids, norm_text, word2ph = M.get_text_for_tts_infer(
        text, language, symbol_to_id=symbol_to_id)
    z_p, pronoun_lens, audio_len = sess_enc.run(None, input_feed={
        "phone": phones, "g": g, "tone": tones, "language": lang_ids,
        "noise_scale": np.array([0], dtype=np.float32),
        "length_scale": np.array([1.0 / speed], dtype=np.float32),
        "noise_scale_w": np.array([0], dtype=np.float32),
        "sdp_ratio": np.array([0], dtype=np.float32)})
    return z_p, pronoun_lens, int(audio_len[0]), word2ph


def _fit_pieces(text, enc, max_samples, depth=3):
    """把一段文本切到"每段预测时长 ≤ max_samples"，返回 [(文本, encoder输出)]。

    判据用 encoder 自己预测的 audio_len，而非猜词数阈值 —— 精确且随语速/语言自适应。
    超限则按词二分递归；词数太少或递归到底就放行（避免无限切分）。
    encoder 是 CPU ONNX，额外几次调用相对 NPU 解码可忽略。
    """
    z_p, pronoun_lens, audio_len, word2ph = enc(text)
    words = text.split()
    if audio_len <= max_samples or len(words) < 4 or depth == 0:
        return [(text, (z_p, pronoun_lens, audio_len, word2ph))]
    mid = len(words) // 2
    return (_fit_pieces(" ".join(words[:mid]), enc, max_samples, depth - 1)
            + _fit_pieces(" ".join(words[mid:]), enc, max_samples, depth - 1))


def synthesize_once(sess_enc, sess_dec, g, symbol_to_id, language, speed, dec_len, text,
                    sr=44100):
    sens = M.split_sentences_into_pieces(text, language, quiet=True)
    enc = lambda t: _encode(sess_enc, g, symbol_to_id, language, speed, t)
    max_samples = int(MAX_SEG_SEC * sr)
    audio_list = []
    for se in sens:
        for _sub, (z_p, pronoun_lens, audio_len, word2ph) in _fit_pieces(se, enc, max_samples):
            word2pronoun = M.calc_word2pronoun(word2ph, pronoun_lens)
            pn_slices, zp_slices = M.generate_slices(word2pronoun, dec_len)
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
    return M.audio_numpy_concat(audio_list, sr=sr, speed=speed)


def main():
    import onnxruntime as ort
    import axengine as axe
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TTS_DATASETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.8)
    args = ap.parse_args()
    lang = tts_lang(args.dataset)
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
    print(f">>> melotts/{args.dataset}: 合成 {ok}/{len(rows)} 条 -> {out_dir}  "
          f"热启动RTF(不含加载)={rtf:.4f}")
    # 本脚本载入一次 + warmup 跳首条，故这就是主口径（不含模型加载）
    (out_dir / "_rtf_pure.txt").write_text(f"{rtf:.4f}\n")


if __name__ == "__main__":
    main()
