"""重新生成 main.js 里的 CC-Canto 词典内联块。

用法：python tools/build-inline-dict.py

数据源：插件根目录 dict-yue.json（CC-Canto / Pleco Inc.，CC BY-SA 3.0，见 README）
输出：重写 main.js 中 const INLINE_YUE_DICT_GZ = '...' 这一行。

为什么需要内联：
    Obsidian 社区插件市场只安装 main.js / manifest.json / styles.css 三件套，
    dict-yue.json 不会随市场安装分发；而社区政策禁止插件运行时自行下载依赖。
    因此把词典作为随 release 分发的静态数据（gzip + base64）放进 main.js。

可复现性：gzip level 9 且 mtime=0，同一份 dict-yue.json 每次生成结果完全一致。
改完请用 node --check main.js 与 yue-test/dict-inline-test.js 验证。
"""
import base64
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DICT = os.path.join(ROOT, 'dict-yue.json')
MAIN = os.path.join(ROOT, 'main.js')


def main():
    if not os.path.isfile(DICT):
        sys.exit('找不到 dict-yue.json：' + DICT)
    raw = open(DICT, 'rb').read()
    rows = json.loads(raw.decode('utf-8'))
    if not isinstance(rows, list) or not rows:
        sys.exit('dict-yue.json 不是非空数组')

    b64 = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode('ascii')
    new_const = "const INLINE_YUE_DICT_GZ = '%s';" % b64

    src = open(MAIN, 'r', encoding='utf-8', newline='').read().replace('\r\n', '\n')
    pattern = r"const INLINE_YUE_DICT_GZ = '[^']*';"
    hits = len(re.findall(pattern, src))
    if hits != 1:
        sys.exit('main.js 中 INLINE_YUE_DICT_GZ 声明命中 %d 次，期望 1 次' % hits)
    src = re.sub(pattern, lambda _m: new_const, src, count=1)
    open(MAIN, 'w', encoding='utf-8', newline='\n').write(src)

    print('已写入 %d 条词条；gzip %.2f MB，base64 %.2f MB' % (
        len(rows), len(gzip.compress(raw, 9, mtime=0)) / 1048576, len(b64) / 1048576))
    print('main.js 当前大小 %.2f MB' % (os.path.getsize(MAIN) / 1048576))


if __name__ == '__main__':
    main()
