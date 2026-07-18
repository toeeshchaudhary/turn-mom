#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "$DEST"
rclone copy "r2:chadgpt-data/Bonzo/Clean Redacted" "$DEST/bonzo_clean_redacted" \
  --transfers 16 --progress
for m in "March Transcripts" "April Transcripts" "May Transcripts"; do
  rclone copy "r2:chadgpt-data/$m" "$DEST/transcripts" --transfers 16 --progress
done
echo "bonzo:       $(find "$DEST/bonzo_clean_redacted" -name '*.json' | wc -l) files"
echo "transcripts: $(find "$DEST/transcripts" -name '*.txt'  | wc -l) files"
