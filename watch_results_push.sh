#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results"
CHECKPOINT_DIR="results/checkpoints"
TRAIN_LOG="results/train2.log"
INTERVAL=1200   # 20 min

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Use already-saved git credentials
git config credential.helper store
git config --global credential.helper store

commit_and_push() {
  local msg="$1"

  git add "$CHECKPOINT_DIR" "$TRAIN_LOG" 2>/dev/null || true

  if git diff --cached --quiet; then
    echo "[watcher] No staged changes."
    return
  fi

  git commit -m "$msg"
  git push origin "$BRANCH"
}

watch_epochs() {
  touch "$TRAIN_LOG"

  tail -n 0 -F "$TRAIN_LOG" | while read -r line; do
    if echo "$line" | grep -Eiq "epoch[[:space:]]*[0-9]+|epoch[[:space:]]*[0-9]+/"; then
      epoch="$(echo "$line" | grep -Eio "epoch[[:space:]]*[0-9]+|epoch[[:space:]]*[0-9]+/" | head -1 | grep -Eo "[0-9]+" || true)"

      if [ -n "$epoch" ]; then
        echo "[watcher] Epoch $epoch detected in train.log"
        commit_and_push "Update training log after epoch $epoch"
      else
        echo "[watcher] Epoch detected in train.log"
        commit_and_push "Update training log after epoch"
      fi
    fi
  done
}

watch_checkpoints() {
  mkdir -p "$CHECKPOINT_DIR"

  while true; do
    sleep "$INTERVAL"

    if ! git diff --quiet -- "$CHECKPOINT_DIR" || \
       ! git diff --cached --quiet -- "$CHECKPOINT_DIR" || \
       [ -n "$(git ls-files --others --exclude-standard "$CHECKPOINT_DIR")" ]; then

      echo "[watcher] Checkpoint change detected."
      commit_and_push "Update checkpoints"
    else
      echo "[watcher] No checkpoint changes."
    fi
  done
}

echo "[watcher] Watching $CHECKPOINT_DIR every 20 min"
echo "[watcher] Watching $TRAIN_LOG for epoch updates"

watch_epochs &
watch_checkpoints &

wait
