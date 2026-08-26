#!/usr/bin/env python3
"""SE / TTS 已适配模型注册表：描述每个模型的本地目录、采样率、类型、语言，
以及「处理单条」的命令构建器（返回 workdir + argv，供 enhance_batch/synth_batch 调用）。
命令依据各模型仓库的 demo/run 脚本编写（板端 axengine 运行）。

chip: ax650 / ax630c / ax620q（映射各模型的 axmodel 变体）。
"""
from pathlib import Path

# ---------------- SE：输入 noisy wav -> 输出 enhanced wav ----------------

_GTCRN_AXMODEL = {"ax650": "gtcrn_650.axmodel", "ax630c": "gtcrn_630.axmodel",
                  "ax620q": "gtcrn_615.axmodel"}


def se_gtcrn(model_dir, chip, inp, outp):
    axm = _GTCRN_AXMODEL.get(chip, "gtcrn_650.axmodel")
    return str(model_dir), ["python", "demo_gtcrn_ax.py", "--model", f"models/{axm}",
                            "--input", str(inp), "--output", str(outp)]


def se_fastenhancer(model_dir, chip, inp, outp):
    # demo.py 在 python/ 下，模型 ../models/16k/model.axmodel（VoiceBank 为 16k）
    return str(Path(model_dir) / "python"), ["python", "demo.py",
            "--model", "../models/16k/model.axmodel", "--input", str(inp), "--output", str(outp)]


# ---------------- TTS：输入 text -> 输出 wav ----------------
# 预设音色（preset）：直接 text->wav；零样本（zeroshot）：用仓库自带固定 prompt。

_KOKORO_VOICE = {"zh": ("z", "checkpoints/voices/zf_xiaoyi.pt"),
                 "en": ("a", "checkpoints/voices/af_heart.pt")}


def tts_kokoro(model_dir, chip, lang, text, outp):
    lc, voice = _KOKORO_VOICE.get(lang, _KOKORO_VOICE["en"])
    mdir = "models_620E" if chip in ("ax630c", "ax620q") else "models"
    return str(model_dir), ["python", "demo_kokoro_ax.py", "--text", text, "--lang", lc,
                            "--voice", voice, "--output", str(outp), "-d", mdir, "-f", "0.3"]


_MELO = {"zh": ("ZH", "zh"), "en": ("EN", "en")}


def tts_melotts(model_dir, chip, lang, text, outp):
    mlang, suf = _MELO.get(lang, _MELO["en"])
    dec = "decoder-ax630c" if chip in ("ax630c", "ax620q") else "decoder-ax650"
    return str(Path(model_dir) / "python"), ["python", "melotts.py", "--sentence", text,
            "--wav", str(outp), "--encoder", f"../encoder-onnx/encoder-{suf}.onnx",
            "--decoder", f"../{dec}/decoder-{suf}.axmodel", "--language", mlang,
            "--sample_rate", "44100"]


# 零样本固定 prompt（取自各仓库 run 脚本）
_ZIPVOICE_PROMPT = {
    "zh": ("assets/moss_prompts/zh_1_4p5s.wav", "不管怎么样我和汤姆还是要感谢贝尔卡金的援手"),
    "en": ("assets/moss_prompts/en_4_4p5s.wav", "This is almost twice the current industry production level per train."),
}
_ZIPVOICE_BIN = {"ax650": "bin/zipvoice_ax650", "ax630c": "bin/zipvoice_ax630c"}


def tts_zipvoice(model_dir, chip, lang, text, outp):
    pwav, ptext = _ZIPVOICE_PROMPT.get(lang, _ZIPVOICE_PROMPT["en"])
    bin_ = _ZIPVOICE_BIN.get(chip, "bin/zipvoice_ax650")
    repo = [] if lang == "zh" else ["--repo-dir", "."]
    return str(model_dir), [bin_, "--model-dir", "models/zipvoice_distill_ax650",
            "--token-file", "resources/zipvoice_hf/zipvoice/tokens.txt",
            "--prompt-wav", pwav, "--prompt-text", ptext, "--text", text,
            "--vocoder-model", "models/vocoder/vocos_full.axmodel",
            "--output-wav", str(outp), "--seed", "42"] + repo


def tts_cosyvoice2(model_dir, chip, lang, text, outp):
    # C++ LLM 二进制，需 tokenizer server(127.0.0.1:12345) + prompt_files；逐条重载，较重
    # 板端建议拷贝到本地盘运行（NFS 读仅 2.7MB/s 会拖慢 embed 加载）：
    #   COSYVOICE2_DIR=/root/cosyvoice2_local bash tts/run.sh
    import os
    local = os.environ.get("COSYVOICE2_DIR")
    if local and Path(local).exists():
        model_dir = Path(local)
    return str(model_dir), ["./main_ax650",
            "--template_filename_axmodel", "CosyVoice-BlankEN-Ax650-prefill_512/qwen2_p128_l%d_together.axmodel",
            "--token2wav_axmodel_dir", "token2wav-axmodels/", "--n_timesteps", "1",
            "--axmodel_num", "24", "--bos", "0", "--eos", "0",
            "--filename_tokenizer_model", "http://127.0.0.1:12345",
            "--filename_post_axmodel", "CosyVoice-BlankEN-Ax650-prefill_512/qwen2_post.axmodel",
            "--filename_decoder_axmodel", "CosyVoice-BlankEN-Ax650-prefill_512/llm_decoder.axmodel",
            "--filename_tokens_embed", "CosyVoice-BlankEN-Ax650-prefill_512/model.embed_tokens.weight.bfloat16.bin",
            "--filename_llm_embed", "CosyVoice-BlankEN-Ax650-prefill_512/llm.llm_embedding.float16.bin",
            "--filename_speech_embed", "CosyVoice-BlankEN-Ax650-prefill_512/llm.speech_embedding.float16.bin",
            "--continue", "0", "--prompt_files", "prompt_files", "--text", text]
    # 注：main_ax650 输出 output*.wav，driver 需另行改名到 outp（见 synth_batch）


# name -> {kind, sr, type, langs, out_glob(可选), builder}
REGISTRY = {
    # SE
    "gtcrn":        {"kind": "se", "sr": 16000, "builder": se_gtcrn},
    "fastenhancer": {"kind": "se", "sr": 16000, "builder": se_fastenhancer},
    # TTS
    "kokoro":       {"kind": "tts", "sr": 24000, "type": "preset",   "langs": ["zh", "en"], "builder": tts_kokoro},
    "melotts":      {"kind": "tts", "sr": 44100, "type": "preset",   "langs": ["zh", "en"], "builder": tts_melotts},
    "zipvoice":     {"kind": "tts", "sr": 24000, "type": "zeroshot", "langs": ["zh", "en"], "builder": tts_zipvoice},
    "cosyvoice2":   {"kind": "tts", "sr": 24000, "type": "cpp",      "langs": ["zh", "en"], "builder": tts_cosyvoice2,
                     "out_glob": "output*.wav", "heavy": True},
}
