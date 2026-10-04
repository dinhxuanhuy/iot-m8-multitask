#!/usr/bin/env bash
# Đồng bộ mọi kết quả khối mở rộng đã có trên VM Colab về results/checkpoints_ext/.
# Chạy định kỳ trong lúc lô đang chạy để không mất tiến độ nếu phiên Colab bị ngắt.
#
#   bash .colab-bundle/sync_results.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export CH="$PWD/.colab-home"
COLAB="$PWD/.venv-build/bin/colab"
mkdir -p results/checkpoints_ext

list=$(HOME="$CH" "$COLAB" ls -s m8 /content/m8run/results/checkpoints_ext 2>/dev/null \
       | grep -E '\.json$' || true)
if [ -z "$list" ]; then
  echo "  (chưa có tệp json nào trên VM)"
  exit 0
fi

new=0
for remote in $list; do
  name=$(basename "$remote")
  if [ -f "results/checkpoints_ext/$name" ]; then
    continue
  fi
  if HOME="$CH" "$COLAB" download -s m8 \
        "/content/m8run/results/checkpoints_ext/$name" \
        "results/checkpoints_ext/$name" >/dev/null 2>&1; then
    echo "  + tải $name"
    new=$((new + 1))
  else
    echo "  ! lỗi tải $name"
  fi
done

echo "  -> đã tải thêm $new tệp; hiện có $(ls -1 results/checkpoints_ext/*.json 2>/dev/null | wc -l)/12 lượt"
