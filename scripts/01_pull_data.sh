#!/usr/bin/env bash
# Pull the training sources from R2 (chadgpt-data). Run on the GPU box, NOT locally
# (21k Bonzo files). Assumes the `r2` rclone remote is configured (see README).
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "$DEST"

# Bonzo: prefer the PII-redacted variant for a customer-facing model.
rclone copy "r2:chadgpt-data/Bonzo/Clean Redacted" "$DEST/bonzo_clean_redacted" \
  --transfers 16 --progress

# Call transcripts (all months) — used for flow/stage coverage, not voice.
for m in "March Transcripts" "April Transcripts" "May Transcripts"; do
  rclone copy "r2:chadgpt-data/$m" "$DEST/transcripts" --transfers 16 --progress
done

echo "bonzo:       $(find "$DEST/bonzo_clean_redacted" -name '*.json' | wc -l) files"
echo "transcripts: $(find "$DEST/transcripts" -name '*.txt'  | wc -l) files"
