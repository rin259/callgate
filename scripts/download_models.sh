#!/usr/bin/env bash
# Download the model sets callgate uses. Nothing is bundled in the git repo:
# the weights are Apache-2.0 but 100+ MB, so fetch them where you need them.
#
#   bash scripts/download_models.sh [models_dir]
set -euo pipefail

ROOT="${1:-models}"
VERSION="v0.1.1"
mkdir -p "$ROOT"

echo "==> FireRedVAD ONNX (VAD, Apache-2.0)"
# Converted from FireRedTeam/FireRedVAD (PyTorch) to ONNX; see NOTICE.
mkdir -p "$ROOT/fireredvad"
curl -L --fail -o /tmp/fireredvad-onnx.tar.bz2 \
  "https://github.com/rin259/callgate/releases/download/$VERSION/fireredvad-onnx.tar.bz2"
tar -xjf /tmp/fireredvad-onnx.tar.bz2 -C "$ROOT/fireredvad"

echo "==> VoxLock (target-speaker isolation, Apache-2.0, via sherpa-onnx)"
curl -L --fail -o /tmp/sherpa-onnx-voxlock-int8.tar.bz2 \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2"
tar -xjf /tmp/sherpa-onnx-voxlock-int8.tar.bz2 -C "$ROOT"

echo "==> TurnSense (end-of-turn, Apache-2.0)"
mkdir -p "$ROOT/turnsense"
HUB="https://huggingface.co/brgroup/TurnSense/resolve/main/TurnSense1.1"
curl -L --fail -o "$ROOT/turnsense/TurnSense1_1_model_int8.onnx" "$HUB/model_int8.onnx"
curl -L --fail -o "$ROOT/turnsense/TurnSense1_1_am.mvn" "$HUB/am.mvn"

echo
echo "Done. Try:"
echo "  python examples/replay_demo.py --models $ROOT \\"
echo "    --wav $ROOT/sherpa-onnx-voxlock-int8/test_wavs/zh-combined.wav \\"
echo "    --voice $ROOT/sherpa-onnx-voxlock-int8/test_wavs/zh-voice-sample-f.wav \\"
echo "    --turnsense $ROOT/turnsense"
