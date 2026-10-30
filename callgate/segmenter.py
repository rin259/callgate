"""Utterance segmentation with a rising noise floor (minimum statistics).

A fixed RMS threshold fails in two opposite ways:

* a quiet room makes the threshold far too low, so every click starts a turn;
* a noisy room makes it far too high, so the caller disappears.

The fix is a floor that tracks the *quiet part* of the last few seconds. The
20th percentile is used rather than the mean, because the mean is dragged up by
the speech we are trying to detect. The floor must be allowed to rise as well as
fall: an implementation that only lowers it ("if rms < noise: noise = ...")
gets stuck at the quietest moment of the call and then treats the whole noisy
restaurant as speech.

Usage (20 ms frames, matching `parec --rate=16000 --channels=1`):

    seg = AdaptiveSegmenter()
    for frame in frames:
        turn = seg.push(frame)
        if turn:
            handle(turn.pcm)
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class Turn:
    pcm: bytes
    start_ms: int
    end_ms: int
    speech_ms: int
    peak_rms: float

    @property
    def dur_sec(self) -> float:
        return len(self.pcm) / 2 / 16000


class AdaptiveSegmenter:
    """Energy VAD + adaptive noise floor + utterance assembly.

    Parameters mirror the knobs that actually mattered in production:

    ``speech_rms``
        Absolute floor for very quiet lines (the client often sends digital
        silence between talkspurts, which would otherwise make the adaptive
        floor zero).
    ``noise_mult``
        How far above the noise floor a frame must be to count as speech.
    ``noise_window_ms``
        Window for the 20th-percentile floor.
    ``end_silence_ms``
        Trailing silence that closes a turn. Keep it small (150-250 ms) when a
        semantic end-of-turn model runs afterwards; that model decides whether
        the user is really done.
    """

    def __init__(self, *, sample_rate: int = 16000, frame_ms: int = 20,
                 speech_rms: float = 380.0, noise_mult: float = 3.0,
                 noise_window_ms: int = 3000, min_speech_ms: int = 220,
                 end_silence_ms: int = 180, max_utter_ms: int = 15000,
                 min_tail_ms: int = 600):
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_bytes = self.sample_rate * 2 * self.frame_ms // 1000
        self.speech_rms = float(speech_rms)
        self.noise_mult = float(noise_mult)
        self.noise_window_ms = int(noise_window_ms)
        self.min_speech_ms = int(min_speech_ms)
        self.end_silence_ms = int(end_silence_ms)
        self.max_utter_ms = int(max_utter_ms)
        self.min_tail_ms = int(min_tail_ms)
        self._win = max(25, self.noise_window_ms // self.frame_ms)
        self.reset()

    def reset(self) -> None:
        self._hist: collections.deque[float] = collections.deque(maxlen=self._win)
        self._noise = 200.0
        self._buf = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self._pos_ms = 0
        self._turn_start_ms = 0
        self._peak = 0.0

    @property
    def noise_floor(self) -> float:
        return self._noise

    def push(self, frame: bytes) -> Turn | None:
        if len(frame) != self.frame_bytes:
            raise ValueError(f"expected {self.frame_bytes} bytes, got {len(frame)}")
        x = np.frombuffer(frame, np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        self._hist.append(rms)
        if len(self._hist) >= 25:
            self._noise = float(np.percentile(np.fromiter(self._hist, np.float32), 20))
        threshold = max(self.speech_rms, self._noise * self.noise_mult)
        voiced = rms > threshold

        if voiced:
            if self._speech_ms == 0:
                self._turn_start_ms = self._pos_ms
            self._speech_ms += self.frame_ms
            self._silence_ms = 0
            self._buf += frame
            self._peak = max(self._peak, rms)
            self._pos_ms += self.frame_ms
            if self._speech_ms >= self.max_utter_ms:
                return self._emit()
            return None

        self._silence_ms += self.frame_ms
        self._pos_ms += self.frame_ms
        if self._speech_ms:
            self._buf += frame                      # keep a little tail
        # A single click never reaches min_speech_ms; drop it instead of letting
        # the buffer grow forever.
        if 0 < self._speech_ms < self.min_speech_ms and self._silence_ms >= self.min_tail_ms:
            self._reset_turn()
            return None
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.end_silence_ms:
            return self._emit()
        return None

    # ── internals ─────────────────────────────────────────────────────────
    def _emit(self) -> Turn:
        turn = Turn(
            pcm=bytes(self._buf),
            start_ms=self._turn_start_ms,
            end_ms=self._pos_ms,
            speech_ms=self._speech_ms,
            peak_rms=self._peak,
        )
        self._reset_turn()
        return turn

    def _reset_turn(self) -> None:
        self._buf = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self._peak = 0.0
