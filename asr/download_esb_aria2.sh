#!/bin/bash
# ============================================================
# ESB 测试数据 aria2c 下载脚本 v2（test-only，可断点续传、可重复运行）
#
# 修复内容:
#   - --header 参数正确引号（v1 的 bug 导致 gated 仓库报 "Unsupported protocol: Bearer"）
#   - 每个文件单独 aria2c 调用 + 显式 -o 文件名（避免 CDN 重定向哈希名/range 错乱）
#   - --max-tries/--retry-wait 自动重试
#   - 结束前校验每个文件是否存在，列出缺失清单
# ============================================================
set -u

BASE=/data/shared/huyuan/8860_datasets/asr
MIRROR=https://hf-mirror.com/datasets
# hf-mirror/CDN 连接数不宜过高；openslr 用 -x16 提速
OPTS="-x8 -s8 -k1M -c --auto-file-renaming=false --console-log-level=warn --summary-interval=30 --max-tries=10 --retry-wait=10"
OPENSLR_OPTS="-x16 -s16 -k1M -c --auto-file-renaming=false --console-log-level=warn --summary-interval=30 --max-tries=10 --retry-wait=10"

if ! command -v aria2c >/dev/null 2>&1; then
    echo "!!! 未安装 aria2c，请先执行: conda install -n base -c conda-forge -y aria2"
    exit 1
fi

# gated 数据集（gigaspeech/spgispeech）需要 HF token，且账号已在网页同意条款
HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || true)"
AUTH=(--header="Authorization: Bearer ${HF_TOKEN}")
[ -n "$HF_TOKEN" ] && echo ">>> 已读取 HF token" || echo ">>> 未找到 HF token（gigaspeech/spgispeech 将失败）"

mkdir -p "$BASE"

dl() {  # dl <url> <输出路径> [额外选项...]
    local url="$1" out="$2"; shift 2
    local dir; dir=$(dirname "$out"); local name; name=$(basename "$out")
    mkdir -p "$dir"
    if aria2c $OPTS "$@" -d "$dir" -o "$name" "$url"; then
        echo "OK:   $out"
    else
        echo "FAIL: $out"
        return 1
    fi
}

# ============================================================
echo ">>> [1/7] librispeech（openslr，-x16）"
dl "http://www.openslr.org/resources/12/test-clean.tar.gz"  "$BASE/librispeech/test-clean.tar.gz"  $OPENSLR_OPTS
dl "http://www.openslr.org/resources/12/test-other.tar.gz"  "$BASE/librispeech/test-other.tar.gz"  $OPENSLR_OPTS

echo ">>> [2/7] common_voice 跳过（30GB 整包 + API 需登录）"

echo ">>> [3/7] voxpopuli"
dl "$MIRROR/polinaeterna/voxpopuli/resolve/main/en/test-00000-of-00001.parquet" "$BASE/voxpopuli/test-00000-of-00001.parquet"

echo ">>> [4/7] tedlium"
dl "$MIRROR/AudioLLMs/tedlium3_test/resolve/main/data/test-00000-of-00001.parquet" "$BASE/tedlium/test-00000-of-00001.parquet"

echo ">>> [5/7] gigaspeech（gated）"
for i in 0 1 2; do
    dl "$MIRROR/speechcolab/gigaspeech/resolve/main/data/audio/test_files/test_chunks_000$i.tar.gz" \
       "$BASE/gigaspeech/audio/test_chunks_000$i.tar.gz" "${AUTH[@]}"
    dl "$MIRROR/speechcolab/gigaspeech/resolve/main/data/metadata/test_metadata/test_chunks_000${i}_metadata.csv" \
       "$BASE/gigaspeech/meta/test_chunks_000${i}_metadata.csv" "${AUTH[@]}"
done

echo ">>> [6/7] spgispeech（gated）"
for i in 0 1 2; do
    dl "$MIRROR/kensho/spgispeech/resolve/main/data/audio/test/test_part_$i.tar.gz" \
       "$BASE/spgispeech/audio/test_part_$i.tar.gz" "${AUTH[@]}"
done
dl "$MIRROR/kensho/spgispeech/resolve/main/data/meta/test.csv" "$BASE/spgispeech/meta/test.csv" "${AUTH[@]}"

echo ">>> [7/7] earnings22"
dl "$MIRROR/anton-l/earnings22_baseline_5_gram/resolve/main/metadata.csv" "$BASE/earnings22/metadata.csv"
for id in 4432298 4450488 4470290 4479741 4483338 4485244; do
    dl "$MIRROR/anton-l/earnings22_baseline_5_gram/resolve/main/data/chunked/$id.tar.gz" \
       "$BASE/earnings22/chunked/$id.tar.gz"
done

echo ">>> [8/7] ami"
for id in EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d \
          IS1009a IS1009b IS1009c IS1009d TS3003a TS3003b TS3003c TS3003d; do
    dl "$MIRROR/speech-seq2seq/ami/resolve/main/audio/ihm/eval/$id.tar.gz" \
       "$BASE/ami/audio/$id.tar.gz"
done
dl "$MIRROR/speech-seq2seq/ami/resolve/main/annotations/eval/text" "$BASE/ami/annotations/text"

# ============================================================
# 校验
# ============================================================
echo ""
echo "================ 校验 ================"
MISSING=0
check() {  # check <文件路径>
    if [ -s "$1" ]; then
        echo "  ✓ $(basename "$1")  ($(du -h "$1" | cut -f1))"
    else
        echo "  ✗ 缺失: $1"
        MISSING=$((MISSING+1))
    fi
}
for f in "$BASE"/librispeech/test-clean.tar.gz "$BASE"/librispeech/test-other.tar.gz \
         "$BASE"/voxpopuli/test-00000-of-00001.parquet \
         "$BASE"/tedlium/test-00000-of-00001.parquet \
         "$BASE"/gigaspeech/audio/test_chunks_000{0,1,2}.tar.gz \
         "$BASE"/gigaspeech/meta/test_chunks_000{0,1,2}_metadata.csv \
         "$BASE"/spgispeech/audio/test_part_{0,1,2}.tar.gz "$BASE"/spgispeech/meta/test.csv \
         "$BASE"/earnings22/metadata.csv \
         "$BASE"/earnings22/chunked/{4432298,4450488,4470290,4479741,4483338,4485244}.tar.gz \
         "$BASE"/ami/annotations/text \
         "$BASE"/ami/audio/{EN2002a,EN2002b,EN2002c,EN2002d,ES2004a,ES2004b,ES2004c,ES2004d,IS1009a,IS1009b,IS1009c,IS1009d,TS3003a,TS3003b,TS3003c,TS3003d}.tar.gz; do
    check "$f"
done
echo "====================================="
if [ "$MISSING" -eq 0 ]; then
    echo ">>> 全部文件下载完成 ✓"
else
    echo ">>> 仍有 $MISSING 个文件缺失，重新运行本脚本即可续传补齐"
fi
