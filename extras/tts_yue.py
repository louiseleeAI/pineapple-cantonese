"""粤语 TTS：edge-tts（微软神经网络音色，zh-HK）

用法（文字从 stdin 读，避免 Windows 命令行编码/引号问题）：
  echo 文字 | python tts_yue.py <输出.mp3> [音色] [语速]

音色：
  zh-HK-HiuGaaiNeural  女声（默认）
  zh-HK-HiuMaanNeural  女声
  zh-HK-WanLungNeural  男声

成功时打印 OK <字节数>，失败打印 ERROR <原因> 并以非 0 退出。
"""
import sys
import os
import asyncio

DEFAULT_VOICE = "zh-HK-HiuGaaiNeural"


async def synth(text, out_path, voice, rate):
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate)
    await comm.save(out_path)


def main():
    if len(sys.argv) < 2:
        print("ERROR 缺少输出路径")
        return 2
    out_path = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else DEFAULT_VOICE
    rate = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "+0%"

    raw = sys.stdin.buffer.read()
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace").strip()
    if not text:
        print("ERROR 没有文本")
        return 2

    try:
        asyncio.run(synth(text, out_path, voice, rate))
    except Exception as e:
        print("ERROR %s: %s" % (type(e).__name__, str(e)[:200]))
        return 3

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
        print("ERROR 合成结果为空")
        return 4

    print("OK %d" % os.path.getsize(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
