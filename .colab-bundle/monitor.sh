#!/usr/bin/env bash
# Giám sát tiến độ: đồng bộ kết quả mỗi 8 phút, in số lượt đã xong.
cd "$(dirname "$0")/.." || exit 1
for i in $(seq 1 90); do
  bash .colab-bundle/sync_results.sh 2>&1 | grep -E '^  \+|hiện có' || true
  n=$(ls -1 results/checkpoints_ext/*.json 2>/dev/null | wc -l)
  cfg=$(ls -1 results/checkpoints_ext/*.json 2>/dev/null | sed 's|.*/||;s|_L10_s.*||' | sort -u | tr '\n' ' ')
  echo "[$(date +%H:%M)] $n/12 lượt  |  cấu hình có kết quả: ${cfg:-chưa có}"
  if [ "$n" -ge 12 ]; then echo "*** ĐỦ 12/12 ***"; break; fi
  sleep 480
done
