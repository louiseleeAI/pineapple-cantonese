"""粤语语音转写：sherpa-onnx + SenseVoice（支持粤语 yue）

用法：
  python transcribe_yue.py <音频> <输出.srt> <模型目录> [线程数]

模型目录需包含 model.int8.onnx 与 tokens.txt。
输出 SRT（带分段时间戳），同时把进度打到 stdout 的 PROGRESS 行。
"""
import sys
import os

# 必须在 import numpy / sherpa_onnx 之前设置：
# 这台机器空闲内存常年不到 2GB，OpenBLAS 会按线程数预分配缓冲，
# 线程一多就在加载模型时 "Memory allocation still failed"。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


def log(msg):
    print(msg, flush=True)


def decode_audio(path, target_sr=16000):
    """用 PyAV 解码任意音频为 16k 单声道 float32"""
    import av

    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            arr = out.to_ndarray()
            chunks.append(arr.reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks)
    return (pcm.astype(np.float32) / 32768.0)



def split_canto_sentences(text):
    """按中文标点分句；英文句点要求后跟空格，避免切坏缩写"""
    import re
    parts = re.split(r'(?<=[。！？；…!?])|(?<=\.)(?=\s)', text)
    out = []
    for raw in parts:
        t = raw.strip()
        if not t:
            continue
        if out and len(out[-1]) <= 2:
            out[-1] += t
        else:
            out.append(t)
    return out



def split_long_clause(start, end, text):
    """过长的一句按逗号切开，但每片至少 8 个字，避免切出「今日，」这种碎片"""
    import re
    raw = [x for x in re.split(r'(?<=[，,；;：:])', text) if x.strip()]
    if len(raw) < 2:
        return [(start, end, text)]
    groups, buf = [], ''
    for x in raw:
        buf += x
        if len(buf) >= 8:
            groups.append(buf)
            buf = ''
    if buf:
        if groups:
            groups[-1] += buf
        else:
            groups.append(buf)
    if len(groups) < 2:
        return [(start, end, text)]
    total = sum(max(1, len(x)) for x in groups) or 1
    out, acc = [], start
    dur = end - start
    for i, x in enumerate(groups):
        d = dur * max(1, len(x)) / total
        seg_end = end if i == len(groups) - 1 else acc + d
        out.append((acc, seg_end, x.strip()))
        acc = seg_end
    return out


def fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    if len(sys.argv) < 4:
        log("用法: transcribe_yue.py <音频> <输出.srt> <模型目录> [线程数]")
        return 2
    audio_path, srt_path, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    model = os.path.join(model_dir, "model.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    if not os.path.exists(model):
        log("ERROR 找不到模型: " + model)
        return 3
    if not os.path.exists(tokens):
        log("ERROR 找不到 tokens.txt: " + tokens)
        return 3

    log("PROGRESS 3  解码音频")
    samples = decode_audio(audio_path)
    if samples.size == 0:
        log("ERROR 音频解码为空")
        return 4
    duration = samples.size / 16000.0
    log(f"时长 {duration:.1f} 秒")

    import sherpa_onnx

    log("PROGRESS 8  加载模型")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model,
        tokens=tokens,
        num_threads=threads,
        use_itn=True,
        language="yue",
        debug=False,
    )

    # VAD 切分长音频，保证每段的起止时间可用
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = os.path.join(model_dir, "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.4
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = 16000

    if not os.path.exists(cfg.silero_vad.model):
        log("WARN 缺 silero_vad.onnx，改为整段识别")
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        cues = [(0.0, duration, text)] if text else []
    else:
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        window = 512
        cues = []
        last_end = 0.0
        pos = 0
        total = samples.size
        while pos < total:
            chunk = samples[pos:pos + window]
            pos += window
            if chunk.size < window:
                chunk = np.pad(chunk, (0, window - chunk.size))
            vad.accept_waveform(chunk)
            while not vad.empty():
                seg = vad.front
                start = seg.start / 16000.0
                dur = len(seg.samples) / 16000.0
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, np.array(seg.samples, dtype=np.float32))
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    cues.append((start, start + dur, text))
                last_end = start + dur
                vad.pop()
            pct = 10 + int(85 * pos / total)
            if pos % (window * 400) == 0:
                log(f"PROGRESS {min(pct, 95)}")
        vad.flush()
        while not vad.empty():
            seg = vad.front
            start = seg.start / 16000.0
            dur = len(seg.samples) / 16000.0
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, np.array(seg.samples, dtype=np.float32))
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                cues.append((start, start + dur, text))
            vad.pop()

    if not cues:
        log("ERROR 没有识别到语音")
        return 5

    # 1) 先合并过短的片段（VAD 偶尔把一句话切碎）
    pieces = []
    for start, end, text in cues:
        if pieces:
            ps, pe, pt = pieces[-1]
            if (start - pe) < 0.45 and len(pt) < 24:
                pieces[-1] = (ps, end, pt + text)
                continue
        pieces.append((start, end, text))

    # 2) 再按标点切成句子级：ASR 的 VAD 段常有十几二十秒，
    #    逐句精听需要更细的粒度，按字数把时间分摊到每句
    merged = []
    for start, end, text in pieces:
        spans = split_canto_sentences(text) or [text]
        total_chars = sum(max(1, len(x)) for x in spans) or 1
        dur = end - start
        acc = start
        for i, sp in enumerate(spans):
            share = max(1, len(sp)) / total_chars
            seg_dur = dur * share
            seg_end = end if i == len(spans) - 1 else acc + seg_dur
            # 单句超过 12 秒（跟读太长）就在逗号处再切一刀
            if seg_end - acc > 12 and ("，" in sp or "," in sp):
                for s2, e2, t2 in split_long_clause(acc, seg_end, sp):
                    merged.append((s2, e2, t2))
            else:
                merged.append((acc, seg_end, sp))
            acc = seg_end

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(merged, 1):
            f.write(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n\n")

    log("PROGRESS 100")
    log(f"DONE {len(merged)} 句 / {duration:.0f} 秒音频")
    return 0


if __name__ == "__main__":
    sys.exit(main())
