# callgate

**Who is talking?** Target-speaker isolation, VAD and turn-taking for phone-call
voice agents — the front-end that decides *whether a turn is worth answering at
all*.

[中文说明](README.zh.md)

```
pip install callgate[all]
```

---

## The failure this fixes

A call from a restaurant or a street carries **other people talking**. Every
component in a typical voice agent assumes the audio is the user:

```
energy VAD  -> "speech"          (it only sees loudness)
speech VAD  -> "speech"          (it sees *someone* speaking)
ASR         -> Chinese words     (someone else's half-sentence)
LLM         -> answers it
```

Measured on a real 72 s restaurant call, the pipeline produced **8 turns** out
of background chatter and bystanders:

| turn | dur | FireRed voiced | speaker cos vs owner |
|---|---|---|---|
| 1 | 1.02s | 0.21s | 0.33 |
| 2 | 1.28s | 0.95s | 0.43 |
| 3 | 0.56s | 0.34s | 0.09 |
| … | | | |

Transcribed as fragments like "个吃好。" "大家。" "868。" — none of it was the
person who made the call. Speaker verification could not save it either: on the
contaminated mixture the real owner scores **0.20–0.40** against their own
profile, overlapping with strangers at 0.20–0.45.

The root cause is upstream of every threshold you are tempted to tune: **the
audio was never separated by speaker.**

## What callgate does

Put a target-speaker extractor *before* the VAD/ASR/embedding chain:

```
mic -> VoxLockIsolator -> AdaptiveSegmenter -> FireRedVAD / TurnSense -> ASR
        (keep the caller)   (find turns)         (is it speech / is it done)
```

| module | what it is | why it matters |
|---|---|---|
| `VoxLockIsolator` | streaming target-speaker extraction (27M causal BSRN) | other voices drop to 1–12% energy; the caller stays at ~100% |
| `AdaptiveSegmenter` | energy VAD with a **rising** 20th-percentile noise floor | a floor that can only fall gets stuck and treats a noisy room as speech |
| `FireRedVAD` | ONNX VAD, numpy + onnxruntime only | no PyTorch in your audio service |
| `TurnSense` | 3-class end-of-turn (`complete/incomplete/invalid`) + decision rule | `invalid` is how noise stops being answered |

## Results

Same segmentation, same thresholds, only the input changed:

| recording | before isolation | after |
|---|---|---|
| restaurant call (bystanders/noise) | **8 turns** | **0 turns** |
| normal call (owner speaking) | 25 turns | **23 turns** |

Speaker cosine against the owner's own profile, same turns:

| | raw phone audio | after isolation |
|---|---|---|
| median | 0.26 | **0.72** |
| range | 0.20–0.40 | 0.48–0.85 |

Isolation cost on a 32-core CPU, 4 threads: **RTF 0.25** at 100 ms chunks
(+100 ms latency, 24 ms algorithmic). Feed it 8 ms chunks and RTF becomes 1.7 —
the fixed per-call overhead dominates.

Full methodology and all tables: [docs/measurements.md](docs/measurements.md).

## Why not just a stronger denoiser?

Because the interference is *speech*. Objective SI-SDR on controlled mixtures:

**caller + environmental noise**

| input SNR | VoxLock | DPDFNet2 |
|---|---|---|
| 10 dB | +5.7 | +6.5 |
| 5 dB | +6.4 | **+9.8** |
| 0 dB | +8.0 | **+12.0** |

**caller + another speaker (interferer 3 dB louder)**

| | VoxLock | DPDFNet2 |
|---|---|---|
| SI-SDR | **+1.8** | **−6.1** |

A generic denoiser wins on stationary noise and actively *hurts* when the noise
is another person: it treats the interferer as speech worth keeping. Chaining
both did not help either (noise: no gain; two speakers: worse).

## Quick start

```bash
git clone https://github.com/rin259/callgate
cd callgate
pip install -e ".[all]"
bash scripts/download_models.sh models
```

Run it on any 16 kHz mono wav, with and without isolation:

```bash
python examples/replay_demo.py --wav call.wav --no-isolate
python examples/replay_demo.py --wav call.wav --voice me.wav
```

Output:

```
#01   3.20s  1.02s  voiced=0.21s  ratio=0.20  peak= 1288  drop (noise)
#02   5.10s  1.28s  voiced=0.95s  ratio=0.74  peak= 744   keep
```

Then measure throughput on your own hardware:

```bash
python examples/bench_isolate.py --wav call.wav --voice me.wav
```

## Using it directly

```python
from callgate import VoxLockIsolator, AdaptiveSegmenter, FireRedVAD

iso = VoxLockIsolator("me.wav", chunk_ms=100, threads=4)
iso.wait_ready()

seg = AdaptiveSegmenter()
vad = FireRedVAD("models/fireredvad/fireredvad_stream_vad_with_cache.onnx",
                 "models/fireredvad/cmvn.ark")

for frame in read_20ms_frames():          # 640 bytes, s16le mono 16 kHz
    frame = iso.push(frame)
    if frame is None:                     # worker still filling its first chunk
        continue
    turn = seg.push(frame)
    if turn and vad.voiced_sec(turn.pcm) >= 0.12:
        handle(turn.pcm)
```

`push()` fails **open**: if the model cannot load, the worker stalls for more
than 2 s, or a chunk raises, you get the original audio back. Isolation is meant
to make the call better, never to mute it.

## What is deliberately not here

- **No QQ / NapCat / SIP adapter.** This is the audio front-end, not a dialer.
  Bring your own transport.
- **No weights, no recordings, no voices.** Apache-2.0 models are downloaded by
  script; every sample in this repo is generated or from upstream test sets.
- **No "answer everything" loop.** `TurnSense` returns `invalid` for noise, and
  `should_respond()` is where you decide to stay silent.

## Limitations

1. Needs ≥3 s of the target voice. Without it the online model enrolls on the
   first 3 s of speech it hears — which on a noisy line may be the waiter.
2. Same-timbre interferers leak: a solo speaker of the same gender survives at
   ~19% energy; still far better than 100%, and embeddings/ASR see much less.
3. The first ~1–3 s of a stream are the least filtered (the model warms up).
4. +100 ms latency, ~0.25 CPU core. Smaller chunks cost more, not less.
5. RTF numbers are from one machine (32-core CPU, no GPU). Measure yours.

## License

Apache-2.0. Model licenses: FireRedVAD Apache-2.0, VoxLock Apache-2.0,
TurnSense Apache-2.0.
