#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required for this installer on macOS."
    exit 1
  fi
  brew install ffmpeg
  brew install --cask musescore
  echo "If the backend cannot find MuseScore, set MUSESCORE_BIN='/Applications/MuseScore 4.app/Contents/MacOS/mscore'."
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg musescore3
else
  echo "Install ffmpeg and MuseScore manually, then set FFMPEG_BIN and MUSESCORE_BIN in .env."
fi

