"""TurnSense — three-class end-of-turn detection (complete / incomplete / invalid).

TurnSense is an open-source semantic turn detector (47M params, INT8 ONNX about
50 MB). Unlike binary "is the user done" models, it also flags **invalid** input
(noise, nonverbal vocalisation), which is what stops a noisy line from being
answered at all.

- Model: https://huggingface.co/brgroup/TurnSense
- Code:  https://github.com/Bairong-Xdynamics/TurnSense (Apache-2.0)

This module reimplements the acoustic frontend (80-dim fbank, LFR 7/6, CMVN)
and the decision rule; it does not depend on their package.

The decision rule matters as much as the model. Using a fixed probability
threshold on ``complete`` cuts far too early, because with three classes the
softmax peak for a finished sentence is often only ~0.58:

    complete   -> cut now
    invalid    -> never cut, and do not answer (noise)
    incomplete -> keep listening until a hard silence cap
"""
from __future__ import annotations

import os
import threading

import numpy as np

LABELS = ("complete", "incomplete", "invalid")


class AudioFrontend:
    """fbank -> LFR -> CMVN, matching the TurnSense training pipeline."""

    def __init__(self, cmvn_file: str, fs: int = 16000, window: str = "hamming",
                 n_mels: int = 80, frame_length: int = 25, frame_shift: int = 10,
                 lfr_m: int = 7, lfr_n: int = 6, dither: float = 0.0):
        import kaldi_native_fbank as knf      # optional dep: only needed for inference

        self._knf = knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False

        self.opts = opts
        self.fs = fs
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn = _load_cmvn(cmvn_file)

    def fbank(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32) * (1 << 15)
        fbank_fn = self._knf.OnlineFbank(self.opts)
        fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        n = fbank_fn.num_frames_ready
        mat = np.empty([n, self.opts.mel_opts.num_bins], dtype=np.float32)
        for i in range(n):
            mat[i, :] = fbank_fn.get_frame(i)
        return mat

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        lfr_inputs = []
        t = inputs.shape[0]
        t_lfr = int(np.ceil(t / lfr_n))
        left = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left, inputs))
        t = t + (lfr_m - 1) // 2
        for i in range(t_lfr):
            if lfr_m <= t - i * lfr_n:
                lfr_inputs.append(inputs[i * lfr_n:i * lfr_n + lfr_m].reshape(1, -1))
            else:
                pad = lfr_m - (t - i * lfr_n)
                frame = inputs[i * lfr_n:].reshape(-1)
                for _ in range(pad):
                    frame = np.hstack((frame, inputs[-1]))
                lfr_inputs.append(frame.reshape(1, -1))
        return np.vstack(lfr_inputs).astype(np.float32)

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars_ = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        return (inputs + means) * vars_

    def extract(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        feat = self.fbank(waveform)
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)
        feat = self.apply_cmvn(feat)
        return feat.astype(np.float32), int(feat.shape[0])


class TurnSense:
    """Three-class end-of-turn model.

    ``predict()`` returns ``(label, {label: prob}, total_prob_of_complete)``.
    """

    def __init__(self, onnx_path: str, cmvn_path: str, num_threads: int = 4):
        import onnxruntime as ort

        for p in (onnx_path, cmvn_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found (see https://huggingface.co/brgroup/TurnSense)")

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.fe = AudioFrontend(cmvn_path)
        self._lock = threading.Lock()
        # Some exports rename the I/O; fall back to positional names.
        self._names = [i.name for i in self.sess.get_inputs()]

    def predict(self, pcm: bytes | np.ndarray, sample_rate: int = 16000,
                max_sec: float = 8.0) -> tuple[str, dict[str, float]]:
        a = _as_float32(pcm, sample_rate)
        n = int(max_sec * sample_rate)
        if len(a) > n:
            a = a[-n:]
        feats, feat_len = self.fe.extract(a)
        feed = {
            self._names[0] if len(self._names) > 0 else "feats":
                feats[None, ...].astype(np.float32),
            self._names[1] if len(self._names) > 1 else "feat_lengths":
                np.array([feat_len], np.int64),
        }
        with self._lock:
            out = self.sess.run(None, feed)[0][0]
        e = np.exp(out - out.max())
        pr = e / e.sum()
        probs = {LABELS[i]: float(pr[i]) for i in range(len(LABELS))}
        return max(probs, key=probs.get), probs


def should_respond(label: str | None, silence_ms: int, *,
                   hard_ms: int = 700, hard_low_ms: int = 1500) -> str | None:
    """Turn a three-class verdict + observed silence into a decision.

    Returns one of:

    ``"respond"``  cut the turn and answer now
    ``"hard"``     model still says "incomplete", but silence is already long
                   enough that waiting looks broken; answer anyway
    ``"skip"``     invalid input (noise / nonverbal) - drop the turn silently
    ``None``       keep listening

    The asymmetry is deliberate: a wrong early cut truncates the user, a wrong
    late cut only costs latency. ``hard_low_ms`` is therefore much larger than
    ``hard_ms``.
    """
    if label == "invalid":
        return "skip" if silence_ms >= hard_ms else None
    if label == "complete":
        return "respond" if silence_ms >= 200 else None
    if silence_ms >= hard_low_ms:
        return "hard"
    if silence_ms >= hard_ms and label is None:
        # model not ready / failed to load: fall back to plain silence timeout
        return "hard"
    return None


def _load_cmvn(path: str) -> np.ndarray:
    means, vars_ = [], []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        item = line.split()
        if not item:
            continue
        if item[0] == "<AddShift>":
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                means = list(nxt[3:len(nxt) - 1])
        elif item[0] == "<Rescale>":
            nxt = lines[i + 1].split()
            if nxt and nxt[0] == "<LearnRateCoef>":
                vars_ = list(nxt[3:len(nxt) - 1])
    return np.array([np.array(means, dtype=np.float64),
                     np.array(vars_, dtype=np.float64)])


def _as_float32(pcm: bytes | np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Return normalised float samples in [-1, 1]."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    a = np.asarray(pcm)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    return a.astype(np.float32)
