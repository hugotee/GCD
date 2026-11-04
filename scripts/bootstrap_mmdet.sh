#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/external/mmdetection"
REV=44ebd17b145c2372c4b700bfb9cb20dbd28ab64a
if [ -d "$TARGET" ]; then
  test "$(git -C "$TARGET" rev-parse HEAD)" = "$REV"
else
  mkdir -p "$ROOT/external"
  git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git "$TARGET"
  test "$(git -C "$TARGET" rev-parse HEAD)" = "$REV"
fi
echo "Pinned official MMDetection: $REV"
echo "Install the validated environment following README.md, then its requirements/runtime.txt."
