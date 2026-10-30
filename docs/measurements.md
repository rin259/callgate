# Measurements

Everything below was measured on one machine: 32-core CPU, RTX 3070 Ti 8 GB
(not used by these models), Python 3.12, onnxruntime/sherpa-onnx CPU builds.
The call recordings are private, so the numbers are reported with the method
that produced them; the scripts in `examples/` reproduce the pipeline on any
wav you supply.

## 1. Turn count before/after isolation

Two real calls, replayed through the same segmenter (`AdaptiveSegmenter`,
`speech_rms=380`, `noise_mult=3.0`, `min_speech_ms=220`, `end_silence_ms=180`):

| recording | length | turns before | turns after isolation |
|---|---|---|---|
| restaurant call (mostly bystanders/noise) | 72.3 s | 8 | **0** |
| normal call (owner speaking) | 461.5 s | 25 | **23** |

The restaurant call's 8 "turns" transcribed to fragments ("个吃好。", "大家。",
"868。", "不发了。", "我。") — none from the account owner. After isolation the
same audio produced no turn at all: the residual energy stayed under the
segmenter's threshold.

The normal call lost 2 short turns (0.6–1.1 s); ASR on the surviving turns was
equal or better:

```
raw                                    isolated
现在几点了？                            现在几点了？
播一段粤语倾听。                         粤语倾听。        (first word clipped)
再说一段韩语。                          再说一段韩语。
好的，那么再把刚才你说的这句话用日。        好的，那么再把刚才你说的这句话用日语。
没事可以继续。                          没事可以继续。
```

## 2. Speaker embedding on raw vs isolated audio

ERes2NetV2 (sherpa-onnx, 192-d) compared against the owner's stored centroid.
Same turns, 0.7 s–3.4 s:

| turn | raw cos | isolated cos |
|---|---|---|
| 0.86 s | 0.34 | 0.73 |
| 0.90 s | −0.18 | 0.67 |
| 1.26 s | 0.81 | 0.79 |
| 1.10 s | 0.31 | 0.77 |
| 3.36 s | 0.63 | 0.82 |
| 1.14 s | −0.12 | 0.66 |
| 1.84 s | −0.15 | 0.85 |
| 2.72 s | −0.13 | 0.79 |
| 1.12 s | 0.08 | 0.57 |

Median 0.26 → 0.72. This is the reason every downstream threshold had to be
loosened: the embeddings were computed on a mixture.

Phone bandwidth alone already costs a lot of cosine:

| condition | cos vs profile |
|---|---|
| reference, full band | 0.874 |
| 8 kHz round-trip | 0.748 |
| Butterworth 300–3400 Hz | 0.532 |
| 8 kHz + band limit | 0.641 |

## 3. Isolation throughput (CPU, 4 threads)

| chunk | RTF |
|---|---|
| 8 ms | 1.73 |
| 32 ms | 0.56 |
| 40 ms | 0.48 |
| **100 ms** | **0.25** |
| 200 ms | 0.20 |

100 ms is the default: it keeps ~4× headroom and adds 100 ms of latency
(24 ms algorithmic inside the model). Smaller chunks are dominated by fixed
per-`run()` overhead.

## 4. Generic denoising vs target-speaker extraction

SI-SDR, aligned (the isolator output is compared after cross-correlation
alignment; without it the numbers are meaningless). Mixtures use the owner's
5.7 s reference, real environmental noise taken from the restaurant call, and a
second speaker's sample.

**caller + noise**

| input SNR | VoxLock | DPDFNet baseline | DPDFNet2 |
|---|---|---|---|
| 20 dB | +1.1 | +0.2 | −1.2 |
| 10 dB | +5.7 | +5.8 | +6.5 |
| 5 dB | +6.4 | +8.4 | **+9.8** |
| 0 dB | +8.0 | +10.9 | **+12.0** |

**caller + interfering speaker (SIR −3 dB)**

| | SI-SDR |
|---|---|
| input | −3.2 |
| VoxLock | **+1.8** |
| DPDFNet2 | −6.1 |
| DPDFNet baseline | −5.0 |

**chained VoxLock → DPDFNet2**

| case | VoxLock | + DPDFNet2 |
|---|---|---|
| 5 dB noise | +11.4 | +11.3 |
| two speakers | +1.8 | −1.4 |

No gain, and worse when the interferer is another person. Not used.

## 5. Speaker selectivity of VoxLock

Upstream female/male test samples, enrolled with one, fed the other:

| enrolled | input | retained energy |
|---|---|---|
| female | female | 107% |
| female | male | 19% |
| male | female | 6% |
| male | male | 102% |
| either | equal-level mix | 69–70% |

Same-timbre interference is the weak spot; the embedding/ASR stages still see a
much cleaner signal than before.

## 6. Reproducing

```bash
# turn counts + per-turn VAD ratio
python examples/replay_demo.py --wav call.wav --no-isolate
python examples/replay_demo.py --wav call.wav --voice me.wav

# throughput on your machine
python examples/bench_isolate.py --wav call.wav --voice me.wav
```

For SI-SDR experiments, mix two wavs at a known SNR/SIR, run `isolate_array()`,
align with cross-correlation, then compute SI-SDR. Note that an unaligned
comparison yields around −30 dB even for a perfect filter.
