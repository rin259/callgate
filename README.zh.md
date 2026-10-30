# callgate

**谁在说话？** 电话语音 Agent 的前端：目标人隔离、VAD、断句与语义端点。

[English](README.md)

```
pip install callgate[all]
```

---

## 它解决的具体问题

餐厅或户外的电话里，**全是别人在说话**。常规语音 Agent 的每个环节都默认
"这段音频就是用户"：

```
能量 VAD   → "有人声"
语音 VAD   → "是人在说"（但不知道是谁）
ASR        → 转出中文（其实是旁人的半句话）
LLM        → 认真回答
```

拿一通 72 秒的真实餐厅通话重放，管线硬生生切出 **8 轮**"用户说话"：

| 轮次 | 时长 | FireRed 人声 | 对本人声纹 cos |
|---|---|---|---|
| 1 | 1.02s | 0.21s | 0.33 |
| 2 | 1.28s | 0.95s | 0.43 |
| 3 | 0.56s | 0.34s | 0.09 |
| … | | | |

转写出来是「个吃好。」「大家。」「868。」——没有一句是打电话的人说的。
声纹也救不了：在混着别人的音频上，**本人对自己的档案只有 0.20–0.40**，
和陌生人的 0.20–0.45 完全重叠。阈值怎么调都是错的。

根因在更前面：**这段音频从来没按人分开过。**

## callgate 做什么

把目标人提取放在 VAD/ASR/声纹**之前**：

```
麦克风 → VoxLockIsolator → AdaptiveSegmenter → FireRedVAD / TurnSense → ASR
         （只留来电人）      （切句）            （是不是人声 / 说完了没）
```

| 模块 | 是什么 | 为什么重要 |
|---|---|---|
| `VoxLockIsolator` | 流式目标人提取（27M 因果 BSRN） | 旁人被压到 1–12% 能量，来电人保留 ~100% |
| `AdaptiveSegmenter` | 底噪取**会升也会降**的 20 分位 | 只降不升的底噪会卡在最低点，整段噪声都算说话 |
| `FireRedVAD` | ONNX VAD，只依赖 numpy + onnxruntime | 音频服务里不用背 PyTorch |
| `TurnSense` | 三分类端点（完整/未完整/无效）+ 判据 | `invalid` 才是"噪声不该被回答"的开关 |

## 实测结果

同一套切句、同一组阈值，只改输入：

| 录音 | 隔离前 | 隔离后 |
|---|---|---|
| 餐厅那通（旁人/环境声） | **8 轮** | **0 轮** |
| 正常通话（本人为主） | 25 轮 | **23 轮** |

同一批轮次，本人声纹余弦：

| | 原始电话音频 | 隔离后 |
|---|---|---|
| 中位数 | 0.26 | **0.72** |
| 范围 | 0.20–0.40 | 0.48–0.85 |

成本（32 核 CPU，4 线程）：**RTF 0.25**，100ms 块，延迟 +100ms。
切成 8ms 块 RTF 会涨到 1.7——每次调用的固定开销占大头。

完整方法与全部表格见 [docs/measurements.md](docs/measurements.md)。

## 为什么不是"换个更强的降噪"

因为干扰本身就是**人声**。受控混合下的 SI-SDR：

**来电人 + 环境噪声**

| 输入 SNR | VoxLock | DPDFNet2 |
|---|---|---|
| 10 dB | +5.7 | +6.5 |
| 5 dB | +6.4 | **+9.8** |
| 0 dB | +8.0 | **+12.0** |

**来电人 + 另一个人说话（对方响 3dB）**

| | VoxLock | DPDFNet2 |
|---|---|---|
| SI-SDR | **+1.8** | **−6.1** |

通用降噪在稳态噪声上更强，但在"另一个人说话"时**帮倒忙**——它把旁人当成
该保留的语音。两者串联也没有收益（噪声无提升，两人场景更差）。

## 快速开始

```bash
git clone https://github.com/rin259/callgate
cd callgate
pip install -e ".[all]"
bash scripts/download_models.sh models
```

拿任意 16kHz 单声道 wav 跑一遍，对比开/关隔离：

```bash
python examples/replay_demo.py --wav call.wav --no-isolate
python examples/replay_demo.py --wav call.wav --voice me.wav
```

```
#01   3.20s  1.02s  voiced=0.21s  ratio=0.20  peak= 1288  drop (noise)
#02   5.10s  1.28s  voiced=0.95s  ratio=0.74  peak= 744   keep
```

测自己机器上的吞吐：

```bash
python examples/bench_isolate.py --wav call.wav --voice me.wav
```

## 直接用

```python
from callgate import VoxLockIsolator, AdaptiveSegmenter, FireRedVAD

iso = VoxLockIsolator("me.wav", chunk_ms=100, threads=4)
iso.wait_ready()

seg = AdaptiveSegmenter()
vad = FireRedVAD("models/fireredvad/fireredvad_stream_vad_with_cache.onnx",
                 "models/fireredvad/cmvn.ark")

for frame in read_20ms_frames():          # 640 字节，s16le 单声道 16kHz
    frame = iso.push(frame)
    if frame is None:                     # 线程还在填第一块
        continue
    turn = seg.push(frame)
    if turn and vad.voiced_sec(turn.pcm) >= 0.12:
        handle(turn.pcm)
```

`push()` **失败即放行**：模型起不来、线程卡住超过 2 秒、某块异常，都会退回
原始音频。隔离是为了让通话更好，不是为了让通话变哑。

## 刻意不包含

- **没有 QQ / NapCat / SIP 适配器**。这是音频前端，不是外呼工具，传输层自己接。
- **不带权重、录音、音色**。Apache-2.0 模型用脚本下载；仓库里没有真人样本。
- **不做"有声音就回答"**。`TurnSense` 对噪声返回 `invalid`，`should_respond()`
  就是决定"闭嘴"的那一行。

## 已知限制

1. 需要目标人 ≥3 秒音色样本。没有样本时在线模型会用听到的前 3 秒自动注册——
   在嘈杂环境里注册到的可能是服务员。
2. 同音色干扰会漏：同性别单独说话仍有约 19% 能量残留（仍远好于 100%，且
   声纹/ASR 看到的东西干净得多）。
3. 每段流的开头 1–3 秒过滤最弱（模型在预热）。
4. +100ms 延迟，约 0.25 核 CPU。块切得更小只会更慢，不会更快。
5. RTF 来自一台机器（32 核，无 GPU），请在你自己的环境复测。

## 许可

Apache-2.0。模型：FireRedVAD Apache-2.0、VoxLock Apache-2.0、TurnSense Apache-2.0。
