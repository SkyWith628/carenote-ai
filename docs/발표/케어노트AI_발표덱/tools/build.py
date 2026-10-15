#!/usr/bin/env python3
"""케어노트 AI 발표 덱 빌드 스크립트.

1) index.html 에 실제로 쓰인 글자만 모아 Gothic A1 / IBM Plex Mono 를 서브셋(woff2)으로 만든다.
   → 오프라인(발표장 와이파이 없음)에서도 같은 글꼴·같은 줄바꿈을 보장.
2) fonts/fonts.css 를 생성한다 (폴더 버전용).
3) 이미지·글꼴을 base64 로 모두 인라인한 단일 파일(standalone)을 만든다.

사용법: python3 tools/build.py            (덱 폴더 안에서 실행; 기본 경로 사용)
       python3 tools/build.py <deck_dir> <fonts_src_dir> <standalone_out.html>
필요: pip install fonttools brotli
"""
import base64, pathlib, re, subprocess, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
deck = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
src = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE / "fonts_src"
out = pathlib.Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else deck.parent / (deck.name + "_standalone.html")

# 글꼴 원본(TTF)이 없으면 Google Fonts 공식 저장소에서 내려받는다 (OFL 라이선스).
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
NEEDED = {
    "GothicA1-Medium.ttf": "gothica1/", "GothicA1-SemiBold.ttf": "gothica1/", "GothicA1-Bold.ttf": "gothica1/",
    "GothicA1-ExtraBold.ttf": "gothica1/", "GothicA1-Black.ttf": "gothica1/", "OFL.txt": "gothica1/",
    "IBMPlexMono-SemiBold.ttf": "ibmplexmono/",
}
src.mkdir(parents=True, exist_ok=True)
for name, folder in NEEDED.items():
    if not (src / name).exists():
        print(f"[download] {name}")
        urllib.request.urlretrieve(GF + folder + name, src / name)
html_path = deck / "index.html"
html = html_path.read_text(encoding="utf-8")

# ---- 1. 글자 수집 ---------------------------------------------------------
chars = set(html)
chars |= set(chr(c) for c in range(0x20, 0x7F))            # ASCII 전체
chars |= set("…·–—‘’“”※→←↔↑↓①②③④⑤⑥⑦⑧⑨⑩%※°±×÷≒≠≤≥∞∴√■□▲△▼▽◆◇○●◎★☆♥♡✓✔")
chars = {c for c in chars if ord(c) >= 0x20 and c not in "­"}
unicodes = ",".join(f"U+{ord(c):04X}" for c in sorted(chars))
print(f"[chars] {len(chars)} distinct code points")

fonts_dir = deck / "fonts"
fonts_dir.mkdir(exist_ok=True)
faces = [  # (family, weight, source file, output name)
    ("Gothic A1", 500, "GothicA1-Medium.ttf", "GothicA1-500.woff2"),
    ("Gothic A1", 600, "GothicA1-SemiBold.ttf", "GothicA1-600.woff2"),
    ("Gothic A1", 700, "GothicA1-Bold.ttf", "GothicA1-700.woff2"),
    ("Gothic A1", 800, "GothicA1-ExtraBold.ttf", "GothicA1-800.woff2"),
    ("Gothic A1", 900, "GothicA1-Black.ttf", "GothicA1-900.woff2"),
    ("IBM Plex Mono", 600, "IBMPlexMono-SemiBold.ttf", "IBMPlexMono-600.woff2"),
]
css_lines = ["/* 서브셋 글꼴: 덱에 쓰인 글자만 포함. 원본 글꼴은 SIL Open Font License (fonts/OFL.txt) */"]
for family, weight, srcname, outname in faces:
    o = fonts_dir / outname
    subprocess.run([
        sys.executable, "-m", "fontTools.subset", str(src / srcname),
        f"--unicodes={unicodes}", "--flavor=woff2", "--layout-features=*",
        "--no-hinting", "--desubroutinize", "--name-IDs=*", f"--output-file={o}",
    ], check=True, capture_output=True)
    print(f"[font] {outname}: {o.stat().st_size // 1024} KB")
    css_lines.append(
        f"@font-face{{font-family:\"{family}\";font-style:normal;font-weight:{weight};font-display:block;"
        f"src:url(\"{outname}\") format(\"woff2\")}}")
(fonts_dir / "fonts.css").write_text("\n".join(css_lines) + "\n", encoding="utf-8")
ofl = src / "OFL.txt"
if ofl.exists():
    (fonts_dir / "OFL.txt").write_text(ofl.read_text(encoding="utf-8"), encoding="utf-8")

# ---- 2. 단일 파일(standalone) ---------------------------------------------
def data_uri(p: pathlib.Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")

inline_css = []
for family, weight, srcname, outname in faces:
    inline_css.append(
        f"@font-face{{font-family:\"{family}\";font-style:normal;font-weight:{weight};font-display:block;"
        f"src:url(\"{data_uri(fonts_dir / outname, 'font/woff2')}\") format(\"woff2\")}}")
standalone = html.replace('<link rel="stylesheet" href="fonts/fonts.css">',
                          "<style>\n" + "\n".join(inline_css) + "\n</style>", 1)
assert standalone != html, "fonts.css link not found"

def repl_img(m):
    name = m.group(1)
    return f'src="{data_uri(deck / "images" / name, "image/webp")}"'
standalone, n = re.subn(r'src="images/([a-z_0-9]+\.webp)"', repl_img, standalone)
print(f"[img] inlined {n} images")
out.write_text(standalone, encoding="utf-8")
print(f"[out] {out} ({out.stat().st_size // 1024} KB)")
