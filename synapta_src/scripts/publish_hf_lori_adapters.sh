#!/bin/zsh
set -euo pipefail

HF_BIN="/Users/uditjain/Library/Python/3.9/bin/hf"
STAGING_ROOT="/Users/uditjain/Desktop/adapter/hf_publish"
USERNAME="uditjain"

REPOS=(
  "lori-qwen2.5-1.5b-math"
  "lori-qwen2.5-1.5b-code"
  "lori-qwen2.5-1.5b-science"
  "lori-qwen2.5-1.5b-legal"
  "lori-qwen2.5-1.5b-medical"
)

if [[ ! -x "$HF_BIN" ]]; then
  echo "hf CLI not found at $HF_BIN"
  exit 1
fi

for repo in "${REPOS[@]}"; do
  local_dir="$STAGING_ROOT/$repo"
  repo_id="$USERNAME/$repo"

  if [[ ! -d "$local_dir" ]]; then
    echo "Missing staging folder: $local_dir"
    exit 1
  fi

  echo "Creating repo: $repo_id"
  "$HF_BIN" repos create "$repo_id" --type model --exist-ok

  echo "Uploading folder: $local_dir -> $repo_id"
  "$HF_BIN" upload "$repo_id" "$local_dir" . --repo-type model \
    --commit-message "Upload LoRI adapter release" \
    --commit-description "Initial public release of LoRI-style PEFT adapter and model card."
done

echo "Done."
