"""callgate - the part of a phone-call voice agent that decides *who* is talking.

Four independent pieces, usable on their own:

    VoxLockIsolator   streaming target-speaker isolation (keeps the caller)
    AdaptiveSegmenter energy VAD with a rising noise floor
    FireRedVAD        ONNX voice activity detection, no PyTorch
    TurnSense         three-class end-of-turn detection

Heavy dependencies (onnxruntime, sherpa-onnx, kaldi-native-fbank) are imported
inside the classes, so importing the package stays cheap.
"""
from __future__ import annotations

__all__ = ["VoxLockIsolator", "isolate_array", "AdaptiveSegmenter", "Turn",
           "FireRedVAD", "TurnSense", "should_respond"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in ("VoxLockIsolator", "isolate_array"):
        from .isolator import VoxLockIsolator, isolate_array
        return {"VoxLockIsolator": VoxLockIsolator, "isolate_array": isolate_array}[name]
    if name in ("AdaptiveSegmenter", "Turn"):
        from .segmenter import AdaptiveSegmenter, Turn
        return {"AdaptiveSegmenter": AdaptiveSegmenter, "Turn": Turn}[name]
    if name == "FireRedVAD":
        from .firered_vad import FireRedVAD
        return FireRedVAD
    if name in ("TurnSense", "should_respond"):
        from .turnsense import TurnSense, should_respond
        return {"TurnSense": TurnSense, "should_respond": should_respond}[name]
    raise AttributeError(name)
