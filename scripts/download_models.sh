#!/usr/bin/env bash
# Download the three model sets callgate uses. Nothing is bundled in the repo:
# the weights are Apache-2.0 but 100+ MB, so fetch them where you need them.
set -euo pipefail

ROOT="${1:-models}"
mkdir -p "$ROOT"

echo "==> FireRedVAD (VAD, Apache-2.0)"
mkdir -p "$ROOT/fireredvad"
BASE="https://huggingface.co/FireRedTeam/FireRedVAD/resolve/main"
curl -L --fail -o "$ROOT/fireredvad/fireredvad_stream_vad_with_cache.onnx" \
  "$BASE/fireredvad_stream_vad_with_cache.onnx"
curl -L --fail -o "$ROOT/fireredvad/cmvn.ark" "$BASE/cmvn.ark"

echo "==> VoxLock (target-speaker isolation, Apache-2.0, via sherpa-onnx)"
mkdir -p "$ROOT/sherpa-onnx-voxlock-int8"
TAG="https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"
curl -L --fail -o "$ROOT/sherpa-onnx-voxlock-int8/chunk_int8.onnx" "$TAG/chunk_int8.onnx"
curl -L --fail -o "$ROOT/sherpa-onnx-voxlock-int8/enroll.onnx" "$TAG/enroll.onnx"

echo "==> TurnSense (end-of-turn, Apache-2.0) - needs huggingface-cli or manual download"
echo "    https://huggingface.co/brgroup/TurnSense"
echo "    Put TurnSense1_1_model_int8.onnx and TurnSense1_1_am.mvn under $ROOT/turnsense/"

echo
echo "Done. Point the examples at --models $ROOT"
