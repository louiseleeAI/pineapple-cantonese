"""跟读评分用的常驻识别服务（粤语）

用法：
  python shadow_server_yue.py <模型目录> [线程数]

协议：stdin 每行一个音频文件路径，stdout 每行一个 JSON。
  {"ok": true, "ready": true}      模型加载完
  {"ok": true, "text": "..."}      识别结果
  {"ok": false, "err": "..."}      这一条失败（进程继续）

单句跟读不需要 VAD 切分，整段直接喂给 SenseVoice 即可。
常驻是为了省掉 sherpa-onnx 每次加载模型的开销（约 3–6 秒）。
"""
import sys
import os
import json

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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
        return None
    import numpy as np
    pcm = np.concatenate(chunks)
    return pcm.astype(np.float32) / 32768.0


def main():
    if len(sys.argv) < 2:
        emit({"ok": False, "err": "缺少模型目录参数"})
        return 2
    model_dir = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    model = os.path.join(model_dir, "model.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    if not os.path.exists(model):
        emit({"ok": False, "err": "找不到模型: %s" % model})
        return 3
    if not os.path.exists(tokens):
        emit({"ok": False, "err": "找不到 tokens.txt: %s" % tokens})
        return 3

    try:
        import sherpa_onnx
    except Exception as e:
        emit({"ok": False, "err": "import sherpa_onnx 失败: %s" % e})
        return 4

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            num_threads=threads,
            use_itn=True,
            language="yue",
            debug=False,
        )
    except Exception as e:
        emit({"ok": False, "err": "加载模型失败: %s" % e})
        return 5

    emit({"ok": True, "ready": True})

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path == "__QUIT__":
            break
        try:
            samples = decode_audio(path)
            if samples is None or samples.size == 0:
                emit({"ok": False, "err": "音频解码为空"})
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, samples)
            recognizer.decode_stream(stream)
            emit({"ok": True, "text": (stream.result.text or "").strip()})
        except Exception as e:
            emit({"ok": False, "err": str(e)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
