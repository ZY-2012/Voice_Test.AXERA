#!/bin/bash
# ============================================================
# ④ 下载已适配模型（axmodel）
# 从 HuggingFace AXERA-TECH 拉取各模型预编译 axmodel 到 model_root/<name>/。
# 芯片由 configs/benchmark.yaml 的 models.chip 决定（ax650/ax630c/ax620q）。
# 注：模型「推理代码」来自各自 GitHub 仓库（见 README 链接），本脚本只取权重。
# 用法:
#   bash download_models.sh              # 全部有 HF 地址的模型
#   bash download_models.sh asr          # 指定模块
# ============================================================
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
MIRROR="$($PY "$REPO_DIR/tools/cfg.py" download.hf_endpoint)"
export HF_ENDPOINT="$MIRROR"
[ $# -gt 0 ] && MODULES=("$@") || MODULES=(asr vad se tts)

# 遍历 config 里各模块模型的 hf 地址并下载
$PY -c "
import sys, os
sys.path.insert(0, '$REPO_DIR/tools')
from common import load_config
cfg = load_config()
model_root = '$MODEL_ROOT'
modules = '${MODULES[*]}'.split()
os.environ['HF_ENDPOINT'] = '$MIRROR'
from huggingface_hub import snapshot_download
for mod in modules:
    models = cfg['models'].get(mod, {})
    for name, spec in models.items():
        hf = (spec or {}).get('hf', '')
        if not hf:
            print(f'  [{mod}] {name}: 无 HF 地址（待适配模型拷入后在 config 填写），跳过')
            continue
        # 目录名优先取 spec.dir（本地目录名与 config 键不一致时，如 tenvad -> ten-vad/）
        dst = os.path.join(model_root, (spec or {}).get('dir') or name)
        print(f'  [{mod}] {name}: 下载 {hf} -> {dst}')
        try:
            snapshot_download(hf, local_dir=dst)
        except Exception as e:
            print(f'    失败: {e}')
"

# 带 code_dir 的模型（推理代码是独立 GitHub 仓库，如 tenvad）：提示手动 clone
$PY -c "
import sys, os
sys.path.insert(0, '$REPO_DIR/tools')
from common import load_config
cfg = load_config()
for mod in '${MODULES[*]}'.split():
    for name, spec in (cfg['models'].get(mod) or {}).items():
        code_dir = (spec or {}).get('code_dir')
        if not code_dir:
            continue
        dst = os.path.join('$MODEL_ROOT', code_dir)
        if os.path.isdir(dst):
            print(f'  [{mod}] {name}: 推理代码已就绪 {dst}')
        else:
            print(f'  [{mod}] {name}: 缺推理代码，请 clone github AXERA-TECH/{code_dir} -> {dst}')
"
echo ">>> 模型下载完成（缺失项为待适配模型）。下一步: bash run_benchmark.sh"
