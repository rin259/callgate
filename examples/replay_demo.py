#!/usr/bin/env python3
"""Replay a wav through callgate and print what a phone agent would see.

    python examples/replay_demo.py --wav call.wav --voice me.wav
    python examples/replay_demo.py --wav call.wav --no-isolate
    python examples/replay_demo.py --wav call.wav --voice me.wav --turnsense models/turnsense

Run it twice on a recording where other people are talking: once with
--no-isolate, once with --voice. That contrast is the point.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from callgate import AdaptiveSegmenter, FireRedVAD, TurnSense, should_respond  # noqa: E402
from callgate.isolator import isolate_array  # noqa: E402

FRAME = 320


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16 kHz mono wav")
    ap.add_argument("--voice", help="enrollment wav (>= 3 s) of the caller")
    ap.add_argument("--no-isolate", action="store_true")
    ap.add_argument("--models", default="models")
    ap.add_argument("--turnsense", help="dir with TurnSense1_1_model_int8.onnx and .mvn")
    ap.add_argument("--rtf-only", action="store_true")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        raise SystemExit(f"need 16 kHz, got {sr}")

    if args.voice and not args.no_isolate:
        print(f"isolating caller voice from {args.voice} ...")
        t0 = time.time()
        audio = isolate_array(
            audio, args.voice,
            model_dir=os.path.join(args.models, "sherpa-onnx-voxlock-int8"))
        dt = time.time() - t0
        sec = max(1e-9, len(audio) / 16000)
        print(f"  {sec:.1f}s audio in {dt:.1f}s  RTF={dt / sec:.2f}\n")
        if args.rtf_only:
            return

    vad = FireRedVAD(os.path.join(args.models, "fireredvad/fireredvad_stream_vad_with_cache.onnx"),
                     os.path.join(args.models, "fireredvad/cmvn.ark"))
    eou = None
    if args.turnsense:
        eou = TurnSense(os.path.join(args.turnsense, "TurnSense1_1_model_int8.onnx"),
                        os.path.join(args.turnsense, "TurnSense1_1_am.mvn"))

    seg = AdaptiveSegmenter()
    turn_n = 0
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        turn = seg.push(audio[i:i + FRAME].tobytes())
        if turn is None:
            continue
        turn_n += 1
        voiced = vad.voiced_sec(turn.pcm)
        dur = len(turn.pcm) / 2 / 16000
        ratio = voiced / max(dur, 1e-6)
        # A turn needs *speech*, not just energy. Short "mm-hmm" still passes:
        # 0.2 s voiced inside 0.25 s is ratio 0.8.
        verdict = "keep" if (voiced >= 0.12 and ratio >= 0.3) else "drop (noise)"
        extra = ""
        if eou is not None and verdict == "keep":
            label, probs = eou.predict(turn.pcm)
            extra = (f"  eou={label}({probs[label]:.2f})"
                     f" -> {should_respond(label, 700) or 'listen'}")
        print(f"#{turn_n:02d} {turn.start_ms/1000:6.2f}s  {dur:4.2f}s  "
              f"voiced={voiced:4.2f}s  ratio={ratio:4.2f}  peak={turn.peak_rms:5.0f}"
              f"  {verdict}{extra}")

    print(f"\n{turn_n} candidate turns. "
          "Compare with --no-isolate on the same file.")


if __name__ == "__main__":
    main()
