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


def se_deepfilternet3(model_dir, chip, inp, outp):
    # 48kHz 模型；inference.py: python inference.py input -o output --model-dir axmodels
    return str(model_dir), ["python", "inference.py", str(inp), "-o", str(outp),
                            "--model-dir", "axmodels"]


def se_gcrn(model_dir, chip, inp, outp):
    # 16kHz；python/python 下 demo.py: --model models/model.axmodel --input X --output Y
    return str(Path(model_dir) / "python"), ["python", "demo.py",
            "--model", "../models/model.axmodel", "--input", str(inp), "--output", str(outp)]


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
# distill 版（4 步采样）；按芯片选目录，勿硬编码——否则 ax630c 会误用 ax650 模型
_ZIPVOICE_MODEL = {"ax650": "models/zipvoice_distill_ax650",
                   "ax630c": "models/zipvoice_distill_ax630C",
                   "ax620q": "models/zipvoice_distill_ax630C"}
# Python 入口的 --model-name（choices 见 infer_zipvoice_axera.py）
_ZIPVOICE_PYMODEL = {"ax650": "zipvoice_distill_ax650",
                     "ax630c": "zipvoice_distill_ax630C",
                     "ax620q": "zipvoice_distill_ax630C"}


def tts_zipvoice(model_dir, chip, lang, text, outp):
    """中文用 C++ 内置 tokenizer；英文必须走 Python daemon（且**不能传 --token-file**）。

    zipvoice.cpp 的 tokenizer 分支是**互斥且有优先级**的：
        if (!cat_tokens_file.empty()) ... else if (tokenizer.IsLoaded()) ... else if (!repo_dir.empty()) ...
    只要 `--token-file` 加载成功就短路走 C++ 内置实现，`--repo-dir` 的 Python daemon 分支
    永远进不去。而 C++ 内置的英文只是 `tokenizer.cpp` 里自述的
    "Simplified English tokenization: character-by-character phoneme mapping" ——
    实测英文回环 WER 95.7%（转写呈"音素沾边但整体错乱"，如 PASTEBOARD→PASTEBARD）。
    官方 run_ax650.sh 英文同时传两个参数，故其英文路径同样落到简化实现；这也解释了
    README 为何只演示 Python 入口 `infer_zipvoice_axera.py`。
    ⇒ 英文只传 --repo-dir，强制走 daemon（daemon 内用 piper_phonemize 做 espeak G2P）。

    英文另需两项前置（见 tts/README.md §3.1/§3.2）：
      1. `cpp/scripts/py_daemon.py` 必须在位（不在默认分发文件列表里）
      2. daemon 依赖 `piper_phonemize`（只出 cp39~cp312 wheel），需 config
         `paths.model_env.zipvoice` 指向 Python≤3.12 的环境；否则 daemon 起不来，
         二进制会**静默降级**到简化 tokenizer（不报错，只是质量崩）。
    中文走 pinyin_table.hpp 查表（编译期烘入），不需要 daemon 与该依赖。
    """
    pwav, ptext = _ZIPVOICE_PROMPT.get(lang, _ZIPVOICE_PROMPT["en"])
    bin_ = _ZIPVOICE_BIN.get(chip, "bin/zipvoice_ax650")
    mdl = _ZIPVOICE_MODEL.get(chip, _ZIPVOICE_MODEL["ax650"])
    argv = [bin_, "--model-dir", mdl,
            "--prompt-wav", pwav, "--prompt-text", ptext, "--text", text,
            "--vocoder-model", "models/vocoder/vocos_full.axmodel",
            "--output-wav", str(outp), "--seed", "42"]
    if lang == "zh":
        argv += ["--token-file", "resources/zipvoice_hf/zipvoice/tokens.txt"]
    else:
        # 不传 --token-file，否则 C++ tokenizer 会短路掉 daemon（见 docstring）
        argv += ["--repo-dir", "."]
    return str(model_dir), argv


def tts_cosyvoice2(model_dir, chip, lang, text, outp):
    # C++ LLM 二进制，需 tokenizer server(127.0.0.1:12345) + prompt_files；逐条重载，较重
    # 板端建议拷贝到本地盘运行（NFS 读仅 2.7MB/s 会拖慢 embed 加载）：
    #   COSYVOICE2_DIR=/root/cosyvoice2_local bash tts/run.sh
    # n_timesteps: flow-matching 去噪步数。vendor 各 run 脚本一律用 1（最快/最低保真），
    # 实测 1 步时有内容截断（del_rate 高、len_ratio<1）。可用环境变量调大做质量/速度权衡：
    #   COSYVOICE2_TIMESTEPS=4 bash tts/run.sh
    import os
    local = os.environ.get("COSYVOICE2_DIR")
    if local and Path(local).exists():
        model_dir = Path(local)
    nts = os.environ.get("COSYVOICE2_TIMESTEPS", "1")
    return str(model_dir), ["./main_ax650",
            "--template_filename_axmodel", "CosyVoice-BlankEN-Ax650-prefill_512/qwen2_p128_l%d_together.axmodel",
            "--token2wav_axmodel_dir", "token2wav-axmodels/", "--n_timesteps", nts,
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
#
# RTF 口径（主指标必须**不含模型加载**，见 README）。逐条起子进程的墙钟时间每条都含加载，
# 不可用作主 RTF。各模型的纯推理耗时来源：
#   rtf_pure_re: 二进制自报计时的正则（对 _run.log 全文匹配，各组求和 = 单条纯推理秒数）
#   batch_driver: 改用"载入一次 + warmup"的批量驱动脚本（python 模型用）
#   两者皆无 -> 该模型拿不到纯 RTF，只写 rtf_e2e（含加载）并在 note 标注
def tts_pocket_tts_zh_en(model_dir, chip, lang, text, outp):
    """零样本（参考音色用包内 Vivian.wav）；单条回退路径。

    批量基准优先走 batch_driver=pocket_tts_zh_en_batch.py（载入一次，纯 RTF 口径）。
    """
    md = Path(model_dir)
    models = md / "models"
    argv = ["python3", "-B", str(md / "board" / "pocket_tts_axera.py"),
            "--text", text, "--reference", str(models / "Vivian.wav"),
            "--output", str(outp),
            "--onnx-dir", str(models), "--axmodel-dir", str(models),
            "--cpu-model-dir", str(models), "--mimi-split-dir", str(models / "mimi_split"),
            "--spm", str(models / "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            "--threads", "4", "--prefill-threads", "8",
            "--npu-flow-net", "1", "--npu-mimi-conv", "1",
            "--flow-ar-model", "flow_ar_step_int8.onnx",
            "--flow-net-model", "flow_net_step_fp32.axmodel",
            "--flow-prefill-model", "flow_step_int8.onnx",
            "--mimi-tf-model", "mimi_transformer_step_int8.onnx",
            "--encoder-dir", str(models / "encoder"),
            "--chunk", "1", "--pause-ms", "120"]
    return str(md), argv


def tts_pocket_tts_zh_en_pinyin(model_dir, chip, lang, text, outp):
    """零样本（参考音色用包内 Vivian.wav）；单条回退路径。

    批量基准优先走 batch_driver=pocket_tts_zh_en_pinyin_batch.py（载入一次，纯 RTF 口径）。
    """
    md = Path(model_dir)
    models = md / "models"
    argv = ["python3", "-B", str(md / "board" / "pocket_tts_axera.py"),
            "--text", text, "--reference", str(models / "Vivian.wav"),
            "--output", str(outp),
            "--onnx-dir", str(models), "--axmodel-dir", str(models),
            "--cpu-model-dir", str(models / "cpu"), "--mimi-split-dir", str(models / "mimi_split"),
            "--tokens-txt", str(models / "tokens.txt"),
            "--bpe-model", str(models / "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            "--threads", "4", "--prefill-threads", "8",
            "--npu-flow-net", "1", "--npu-mimi-conv", "1",
            "--flow-ar-model", "flow_ar_step.onnx",
            "--flow-net-model", "flow_net_step_fp32.axmodel",
            "--flow-prefill-model", "flow_step_windowed.onnx",
            "--mimi-tf-model", "mimi_transformer_step.onnx",
            "--encoder-dir", str(models / "encoder"),
            "--chunk", "1", "--pause-ms", "120"]
    return str(md), argv


REGISTRY = {
    # SE
    "gtcrn":        {"kind": "se", "sr": 16000, "builder": se_gtcrn},
    "fastenhancer": {"kind": "se", "sr": 16000, "builder": se_fastenhancer},
    "deepfilternet3": {"kind": "se", "sr": 48000, "builder": se_deepfilternet3},
    "gcrn":        {"kind": "se", "sr": 16000, "builder": se_gcrn},
    # TTS
    "kokoro":       {"kind": "tts", "sr": 24000, "type": "preset",   "langs": ["zh", "en"], "builder": tts_kokoro},
    "melotts":      {"kind": "tts", "sr": 44100, "type": "preset",   "langs": ["zh", "en"], "builder": tts_melotts,
                     # 载入一次 + warmup 1 条，只计推理（纯 RTF 的正确来源）
                     "batch_driver": "melotts_batch.py"},
    "zipvoice":     {"kind": "tts", "sr": 24000, "type": "zeroshot", "langs": ["zh", "en"], "builder": tts_zipvoice,
                     # 英文需专用 python 环境（Python≤3.12 + piper_phonemize），路径各人不同，
                     # 配在 configs/benchmark.yaml 的 paths.model_env_bin.zipvoice，代码不写死。
                     # 中文(C++)自报 `NPU total` + `Vocoder total`，英文路径同样走 C++ 计时，
                     # 均已排除模型加载。
                     "rtf_pure_re": [r"NPU total:\s*([\d.]+)\s*s",
                                     r"Vocoder total:\s*([\d.]+)\s*s",
                                     r"推理耗时:\s*([\d.]+)s"]},
    "pocket_tts_zh_en": {"kind": "tts", "sr": 24000, "type": "zeroshot", "langs": ["zh", "en"],
                         "builder": tts_pocket_tts_zh_en,
                         # 载入一次 + warmup 1 条，只计推理（纯 RTF 的正确来源）
                         "batch_driver": "pocket_tts_zh_en_batch.py"},
    "pocket_tts_zh_en_pinyin": {"kind": "tts", "sr": 24000, "type": "zeroshot", "langs": ["zh", "en"],
                               "builder": tts_pocket_tts_zh_en_pinyin,
                               # 载入一次 + warmup 1 条，只计推理（纯 RTF 的正确来源）
                               "batch_driver": "pocket_tts_zh_en_pinyin_batch.py"},
    "cosyvoice2":   {"kind": "tts", "sr": 24000, "type": "cpp",      "langs": ["zh", "en"], "builder": tts_cosyvoice2,
                     "out_glob": "output*.wav", "heavy": True},
                     # main_ax650 只打印 decode tokens，不自报耗时 -> 无纯 RTF（仅 rtf_e2e）
}
