#!/usr/bin/env python3
"""Measure throughput and latency of the isolation stage on this machine.

    python examples/bench_isolate.py --voice me.wav --wav call.wav

Reports RTF for several chunk sizes. 100 ms is the default: below that, the
fixed cost of each ONNX ``run()`` dominates and the model stops keeping up with
real time even though its "compute" is small.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from callgate.isolator import VoxLockIsolator  # noqa: E402


def run(audio: np.ndarray, voice: str, chunk_ms: int, threads: int,
        model_dir: str, rounds: int = 1):
    iso = VoxLockIsolator(voice, chunk_ms=chunk_ms, threads=threads,
                          model_dir=model_dir)
    if not iso.wait_ready():
        raise SystemExit(f"isolator failed: {iso.error}")
    frame = 16000 * 20 // 1000
    chunk = 16000 * chunk_ms // 1000
    per_chunk = chunk // frame
    out_frames = 0
    import time
    t0 = time.time()
    for _ in range(rounds):
        for i in range(0, len(audio) - chunk + 1, chunk):
            for j in range(0, chunk, frame):
                iso.push(audio[i + j:i + j + frame].tobytes())
            deadline = time.time() + 2.0
            got = 0
            while got < per_chunk and time.time() < deadline:
                try:
                    iso._out.get(timeout=0.2)
                    got += 1
                except Exception:
                    break
            out_frames += got
    dt = time.time() - t0
    iso.close()
    audio_sec = len(audio) / 16000 * rounds
    return dt, audio_sec, iso.stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    model_dir = os.path.join(args.models, "sherpa-onnx-voxlock-int8")
    audio, sr = sf.read(args.wav, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"{len(audio)/16000:.1f}s of audio\n")
    print(f"{'chunk':>8} {'threads':>8} {'wall':>8} {'RTF':>6}")
    for chunk_ms in (20, 40, 100, 200):
        for threads in (4,):
            dt, sec, st = run(audio, args.voice, chunk_ms, threads, model_dir)
            print(f"{chunk_ms:6d}ms {threads:8d} {dt:7.2f}s {dt/sec:6.2f}")


if __name__ == "__main__":
    main()
