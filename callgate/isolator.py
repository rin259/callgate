"""Streaming target-speaker isolation for phone calls.

Why this exists
---------------
A telephone line in a restaurant or on a street carries **other people talking**.
Energy VADs and speech VADs both say "speech" (they do not know whose), and
speaker embeddings taken on that mixture are useless: measured on real calls,
the caller scores cos 0.20-0.40 against their own profile, overlapping with
strangers at 0.20-0.45. Every downstream threshold then has to be loosened, and
noise gets in.

Generic denoisers do not fix this. Measured on a caller + interfering speaker
mix (SIR -3 dB), DPDFNet2 makes SI-SDR *worse* (-3.2 dB -> -6.1 dB) because it
treats the interferer as speech worth keeping.

VoxLock (27M causal band-split RNN, Apache-2.0) is a target-speaker extractor:
conditioned on a short enrollment sample, it keeps that voice and suppresses the
rest. In the same setup it improves SI-SDR by +4.9 dB.

Measured on a real 72 s restaurant call (private recording, methodology in
docs/measurements.md): 8 turns were produced by noise/other people, 0 after
isolation, while 23 of 25 real caller turns survived on a normal call.

Cost
----
RTF ~0.25 on 4 CPU threads with 100 ms chunks (24 ms algorithmic latency).
Do not feed it 8 ms chunks: per-call overhead pushes RTF to ~1.7.

Models: https://github.com/arbazsiddiqui/voxlock (Apache-2.0), packaged for
sherpa-onnx at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get(
    "VOXLOCK_MODEL_DIR", "models/sherpa-onnx-voxlock-int8")


class VoxLockIsolator:
    """Keep the enrolled speaker, suppress everyone else, in real time.

    Parameters
    ----------
    voice_sample:
        Path to a wav with >= 3 s of the target speaker. Without it the online
        model passes audio through until it has heard 3 s of speech and then
        enrolls on that - which, on a noisy line, may be somebody else.
    sample_rate:
        Input rate. 16000 is what the model expects.
    chunk_ms:
        100 ms is the throughput sweet spot. Smaller chunks cost headroom
        because every ``run()`` call has fixed overhead.
    threads:
        ONNX intra-op threads.

    ``push(frame)`` is non-blocking. It returns a processed frame of the same
    size as the input, or ``None`` while the worker fills its first chunk.
    Callers are expected to skip that tick (~100 ms at startup, then never).
    """

    def __init__(self, voice_sample: str, *, sample_rate: int = 16000,
                 chunk_ms: int = 100, threads: int = 4,
                 model_dir: str | None = None, queue_frames: int = 400):
        self._voice_sample = voice_sample
        self.sample_rate = int(sample_rate)
        self.frame_ms = 20
        self.frame_bytes = self.sample_rate * 2 * self.frame_ms // 1000
        self.chunk_ms = int(chunk_ms)
        self.chunk_bytes = self.sample_rate * 2 * self.chunk_ms // 1000
        self.threads = int(threads)
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.ok = False
        self.error: str | None = None
        self._alive = True
        self._in: queue.Queue[bytes] = queue.Queue(maxsize=queue_frames)
        self._out: queue.Queue[bytes] = queue.Queue(maxsize=queue_frames)
        self._blocks = 0
        self._ms_total = 0.0
        self._broken = 0
        self._last_out = 0.0
        self._sd = None
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    # ── public ────────────────────────────────────────────────────────────
    def wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline and not self.ok and self.error is None:
            time.sleep(0.02)
        return self.ok

    def push(self, frame: bytes) -> bytes | None:
        """Feed one 20 ms frame, get one back (or None while filling)."""
        if not self.ok:
            return frame                       # fail open: never mute the call
        if self._last_out and time.time() - self._last_out > 2.0:
            return frame                       # worker stalled: fail open
        try:
            self._in.put_nowait(frame)
        except queue.Full:
            pass
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def stats(self) -> dict[str, float | int]:
        sec = self._blocks * self.chunk_ms / 1000.0
        return {
            "blocks": self._blocks,
            "audio_sec": round(sec, 1),
            "rtf": round((self._ms_total / 1000.0 / sec), 3) if sec else 0.0,
            "errors": self._broken,
        }

    def close(self) -> None:
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "VoxLockIsolator":
        self.wait_ready()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── worker ────────────────────────────────────────────────────────────
    def _main(self) -> None:
        missing = [os.path.join(self.model_dir, n) for n in ("chunk_int8.onnx", "enroll.onnx")
                   if not os.path.exists(os.path.join(self.model_dir, n))]
        if missing:
            self.error = ("missing model files: " + ", ".join(missing) +
                          " (run scripts/download_models.sh, or pass model_dir=)")
            return
        if not os.path.exists(self._voice_sample):
            self.error = f"voice sample not found: {self._voice_sample}"
            return
        try:
            import sherpa_onnx as so

            cfg = so.OfflineSpeechDenoiserModelConfig(
                voxlock=so.OfflineSpeechDenoiserVoxLockModelConfig(
                    chunk_model=os.path.join(self.model_dir, "chunk_int8.onnx"),
                    enroll_model=os.path.join(self.model_dir, "enroll.onnx"),
                    voice_sample=self._voice_sample,
                ),
                num_threads=self.threads,
            )
            # The online denoiser takes the same model-config object.
            self._sd = so.OnlineSpeechDenoiser(so.OnlineSpeechDenoiserConfig(model=cfg))
            self.ok = True
        except Exception as e:      # pragma: no cover - depends on local setup
            self.error = f"{type(e).__name__}: {e}"
            return

        buf = bytearray()
        while self._alive:
            try:
                buf += self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            while len(buf) >= self.chunk_bytes:
                block = bytes(buf[:self.chunk_bytes])
                del buf[:self.chunk_bytes]
                t0 = time.time()
                try:
                    a = np.frombuffer(block, np.int16).astype(np.float32) / 32768.0
                    out = np.asarray(self._sd.run(a, self.sample_rate).samples)
                    pcm = (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    if len(pcm) < self.chunk_bytes:
                        pcm += b"\x00" * (self.chunk_bytes - len(pcm))
                except Exception:
                    self._broken += 1
                    pcm = block
                self._ms_total += (time.time() - t0) * 1000.0
                self._blocks += 1
                self._last_out = time.time()
                for i in range(0, len(pcm), self.frame_bytes):
                    try:
                        self._out.put_nowait(pcm[i:i + self.frame_bytes])
                    except queue.Full:
                        pass


def isolate_array(pcm: np.ndarray, voice_sample: str, *, sample_rate: int = 16000,
                  chunk_ms: int = 100, threads: int = 4,
                  model_dir: str | None = None) -> np.ndarray:
    """Offline helper: run a whole int16 array through the isolator."""
    iso = VoxLockIsolator(voice_sample, sample_rate=sample_rate, chunk_ms=chunk_ms,
                          threads=threads, model_dir=model_dir)
    if not iso.wait_ready():
        iso.close()
        raise RuntimeError(f"isolator unavailable: {iso.error}")
    frame = sample_rate * iso.frame_ms // 1000
    per_chunk = iso.chunk_bytes // iso.frame_bytes
    out: list[bytes] = []
    for i in range(0, len(pcm) - iso.chunk_bytes + 1, iso.chunk_bytes):
        block = pcm[i:i + iso.chunk_bytes]
        for j in range(0, len(block), frame):
            iso.push(block[j:j + frame].tobytes())
        out.extend(_drain(iso, per_chunk))
    iso.close()
    if not out:
        return np.zeros(0, np.int16)
    return np.frombuffer(b"".join(out), np.int16)


def _drain(iso: VoxLockIsolator, n: int, timeout: float = 2.0) -> list[bytes]:
    got: list[bytes] = []
    deadline = time.time() + timeout
    while len(got) < n and time.time() < deadline:
        try:
            got.append(iso._out.get(timeout=0.2))
        except queue.Empty:
            break
    return got
