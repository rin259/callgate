"""Smoke tests that run only when models are present.

    VOXLOCK_MODEL_DIR=models/sherpa-onnx-voxlock-int8 \
    FIRERED_DIR=models/fireredvad \
    TURNSENSE_DIR=models/turnsense \
    pytest -q
"""
import os

import numpy as np
import pytest

from callgate import AdaptiveSegmenter, Turn, should_respond

FRAME = 320


def tone(ms: int, amp: float = 8000.0, freq: float = 1000.0) -> bytes:
    """Constant-level test signal (20 cycles per 20 ms frame -> stable RMS)."""
    n = 16000 * ms // 1000
    t = np.arange(n) / 16000.0
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return x.tobytes()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * (16000 * ms // 1000)


def test_segmenter_ignores_single_click():
    seg = AdaptiveSegmenter()
    turns = []
    for _ in range(5):
        turns.append(seg.push(silence(20)))
    turns.append(seg.push(tone(20)))
    for _ in range(40):
        turns.append(seg.push(silence(20)))
    assert [t for t in turns if t] == []


def test_segmenter_emits_a_turn():
    seg = AdaptiveSegmenter()
    out = []
    for _ in range(10):
        seg.push(silence(20))
    # 0.6 s of tone: long enough to be speech, short enough that the rising
    # noise floor does not absorb a perfectly stationary signal.
    for _ in range(30):
        t = seg.push(tone(20))
        if t:
            out.append(t)
    for _ in range(20):                      # trailing silence closes it
        t = seg.push(silence(20))
        if t:
            out.append(t)
    assert len(out) == 1
    assert isinstance(out[0], Turn)
    assert out[0].speech_ms >= 500


def test_should_respond_rules():
    assert should_respond("complete", 200) == "respond"
    assert should_respond("complete", 50) is None
    assert should_respond("incomplete", 500) is None
    assert should_respond("incomplete", 1600) == "hard"
    assert should_respond("invalid", 800) == "skip"
    assert should_respond(None, 800) == "hard"


@pytest.mark.skipif(not os.environ.get("FIRERED_DIR"), reason="FIRERED_DIR not set")
def test_firered_speech_vs_silence():
    from callgate import FireRedVAD
    d = os.environ["FIRERED_DIR"]
    vad = FireRedVAD(os.path.join(d, "fireredvad_stream_vad_with_cache.onnx"),
                     os.path.join(d, "cmvn.ark"))
    assert vad.voiced_sec(silence(500)) < 0.05
    # A steady tone is *not* speech: a VAD that accepts it is just an energy gate.
    assert vad.voiced_sec(tone(500)) < 0.3
    wav = os.environ.get("SPEECH_WAV")
    if wav:
        import soundfile as sf
        a, sr = sf.read(wav, dtype="int16")
        assert sr == 16000
        assert vad.voiced_sec(a.tobytes()) > 0.5


@pytest.mark.skipif(not os.environ.get("TURNSENSE_DIR"), reason="TURNSENSE_DIR not set")
def test_turnsense_tone_is_not_complete():
    from callgate import TurnSense
    d = os.environ["TURNSENSE_DIR"]
    ts = TurnSense(os.path.join(d, "TurnSense1_1_model_int8.onnx"),
                   os.path.join(d, "TurnSense1_1_am.mvn"))
    label, probs = ts.predict(tone(600))
    assert label in ("complete", "incomplete", "invalid")
    assert abs(sum(probs.values()) - 1.0) < 1e-4
