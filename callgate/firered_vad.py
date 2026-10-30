"""FireRedVAD (ONNX) — voice activity detection without PyTorch.

The official implementation depends on PyTorch. This is a numpy + onnxruntime
port of the feature pipeline and the streaming forward pass, so it can run in a
small service next to `parec`/`arecord` without dragging in a 2 GB wheel.

Reference implementation: https://github.com/FireRedTeam/FireRedVAD (Apache-2.0)
Models: https://huggingface.co/FireRedTeam/FireRedVAD

Typical use:

    vad = FireRedVAD("fireredvad_stream_vad_with_cache.onnx", "cmvn.ark")
    p = vad.probs(pcm_int16_bytes)          # 10 ms per frame, in [0, 1]
    speech = p >= 0.5
"""
from __future__ import annotations

import math
import os
import threading

import kaldi_native_fbank as knf
import kaldiio
import numpy as np


class CMVN:
    """Kaldi ``.ark`` CMVN stats -> (x - mean) * istd."""

    def __init__(self, path: str):
        stats = kaldiio.load_mat(path)
        assert stats.shape[0] == 2, f"expected 2 rows of CMVN stats, got {stats.shape}"
        self.dim = stats.shape[-1] - 1
        count = float(stats[0, self.dim])
        means, istd = [], []
        for d in range(self.dim):
            mean = float(stats[0, d]) / count
            var = (float(stats[1, d]) / count) - mean * mean
            means.append(mean)
            istd.append(1.0 / math.sqrt(max(var, 1e-20)))
        self.means = np.array(means, dtype=np.float32)
        self.istd = np.array(istd, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.means) * self.istd


class Fbank:
    """80-dim log-mel, 25 ms window / 10 ms hop, snip_edges (same as upstream)."""

    def __init__(self, num_bins: int = 80, sample_rate: int = 16000):
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = sample_rate
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_bins
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(16000, np.asarray(wav, dtype=np.float32).tolist())
        frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        if not frames:
            return np.zeros((0, self.opts.mel_opts.num_bins), dtype=np.float32)
        return np.vstack(frames).astype(np.float32)


class FireRedVAD:
    """Streaming VAD with an internal cache.

    ``probs()`` keeps cache state across calls, so consecutive 20 ms frames can
    be fed one by one. ``voiced_sec()`` resets the cache and evaluates a whole
    utterance in isolation, which is what you want for "is this turn real
    speech" rather than for streaming.
    """

    def __init__(self, onnx_path: str, cmvn_path: str, num_threads: int = 1):
        import onnxruntime as ort

        for p in (onnx_path, cmvn_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found (run scripts/download_models.sh, or pass --models)")

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.cmvn = CMVN(cmvn_path)
        self.fbank = Fbank()
        self.caches = np.zeros((8, 1, 128, 19), dtype=np.float32)
        # The cache is shared state: callers may hit this from an audio thread
        # and a speculative-ASR thread at the same time.
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.caches = np.zeros((8, 1, 128, 19), dtype=np.float32)

    def probs(self, pcm: bytes | np.ndarray) -> np.ndarray:
        """Speech probability per 10 ms frame. Keeps the streaming cache."""
        wav = _as_float32(pcm)
        feat = self.cmvn(self.fbank(wav))[None, ...]
        if feat.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            out = self.sess.run(None, {"feat": feat, "caches_in": self.caches})
            self.caches = out[1]
        return out[0][0, :, 0]

    def voiced_sec(self, pcm: bytes | np.ndarray, threshold: float = 0.5) -> float:
        """Seconds of speech in this utterance (cache reset first)."""
        self.reset()
        p = self.probs(pcm)
        if p.size == 0:
            return 0.0
        return float((p >= threshold).sum()) * 0.01

    def regions(self, pcm: bytes | np.ndarray, threshold: float = 0.5,
                min_gap_frames: int = 5) -> list[tuple[float, float]]:
        """Contiguous speech regions as ``[(start_sec, end_sec), ...]``."""
        p = self.probs(pcm)
        mask = p >= threshold
        out: list[tuple[float, float]] = []
        i, n = 0, len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = last = i
            while j < n:
                if mask[j]:
                    last = j
                elif j - last > min_gap_frames:
                    break
                j += 1
            out.append((i * 0.01, (last + 1) * 0.01))
            i = j
        return out


def _as_float32(pcm: bytes | np.ndarray) -> np.ndarray:
    """Return int16-scale float samples (the scale kaldi fbank expects)."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a = np.asarray(pcm)
    if a.dtype == np.int16:
        return a.astype(np.float32)
    a = a.astype(np.float32)
    if a.size and float(np.max(np.abs(a))) <= 1.5:   # already normalised
        a = a * 32768.0
    return a
