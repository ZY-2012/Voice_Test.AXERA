#!/bin/bash
# ============================================================
# 交叉编译 TEN VAD 板端可执行（在 **host** 端跑，产物走共享挂载给板端用）
# 依赖：cmake ≥3.16 + aarch64-linux-gnu-gcc/g++ + AX650 BSP SDK
#       （SDK 需含 include/ax_engine_api.h、lib/libax_{engine,interpreter,sys}.so）
# 产物：<code_dir>/install/{example/ten_vad_example, aarch64-lib/lib/libten_vad.so}
#       并把模型软链到 <code_dir>/axmodel/（源码硬编码相对路径 axmodel/ten-vad-ax650.axmodel）
# 用法: bash vad/build_tenvad.sh [SDK_ROOT]
# ============================================================
set -eu
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="$($PY "$REPO_DIR/tools/cfg.py" paths.model_root)"
CODE_DIR="$MODEL_ROOT/$($PY "$REPO_DIR/tools/cfg.py" models.vad.tenvad.code_dir)"
MODEL_DIR="$MODEL_ROOT/$($PY "$REPO_DIR/tools/cfg.py" models.vad.tenvad.dir)"
SDK_ROOT="${1:-${AX650_SDK_ROOT:-$($PY "$REPO_DIR/tools/cfg.py" paths.ax650_sdk_root)}}"

[ -d "$CODE_DIR" ] || { echo "!! 缺推理代码 $CODE_DIR（git clone AXERA-TECH/ten-vad.axera）"; exit 1; }
[ -f "$SDK_ROOT/lib/libax_engine.so" ] || {
    echo "!! AX650 SDK 无效: $SDK_ROOT"
    echo "   需含 include/ax_engine_api.h + lib/libax_{engine,interpreter,sys}.so"
    echo "   用法: bash vad/build_tenvad.sh /path/to/ax650n_bsp_sdk/msp/out"
    exit 1; }
command -v aarch64-linux-gnu-gcc >/dev/null || {
    echo "!! 缺交叉编译器: apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"; exit 1; }

echo ">>> 交叉编译 TEN VAD (SDK=$SDK_ROOT)"
cmake -S "$CODE_DIR" -B "$CODE_DIR/build" \
    -DCMAKE_TOOLCHAIN_FILE="$CODE_DIR/aarch64-linux.ini" \
    -DTEN_VAD_SDK_ROOT="$SDK_ROOT" \
    -DCMAKE_INSTALL_PREFIX="$CODE_DIR/install" >/dev/null
cmake --build "$CODE_DIR/build" -j"$(nproc)" >/dev/null
cmake --install "$CODE_DIR/build" >/dev/null

# 源码写死相对路径 axmodel/ten-vad-ax650.axmodel（cwd=code_dir），软链到模型仓库
mkdir -p "$CODE_DIR/axmodel"
AXM="$MODEL_DIR/models/axmodel/ten-vad-ax650.axmodel"
if [ -f "$AXM" ]; then
    ln -sfn "$AXM" "$CODE_DIR/axmodel/ten-vad-ax650.axmodel"
else
    echo "  !! 缺 axmodel: $AXM（跑 bash download_models.sh vad）"
fi

echo ">>> 完成: $CODE_DIR/install/example/ten_vad_example"
ls -l "$CODE_DIR/axmodel/" 2>/dev/null || true
