#!/usr/bin/env python3
"""PDF 페이지 고해상도 렌더(4000×2250)에서 일러스트를 잘라내고 배경을 투명하게 만든다.

- 좌표는 2000×1125 기준(참조 이미지) → 2배.
- 배경 제거: 테두리에서 흘러들어오는 flood fill (흰색·말풍선 하늘색) → 알파 0.
- 작은 연결 요소(점선 조각·꽃·따옴표 등) 제거: min_area 미만 성분 삭제.
"""
import sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = pathlib.Path(sys.argv[1])   # pages/
OUT = pathlib.Path(sys.argv[2])   # deck/images/
OUT.mkdir(parents=True, exist_ok=True)

# name: (page, (x0,y0,x1,y1) in 2000-space, bg colors, min_area, max_dim)
SPEC = {
    "hero":    (1,  (1250, 370, 1850, 1000), [(255,255,255)], 3000, 1200),
    "agent":   (3,  (270, 280, 730, 745),    [(255,255,255)], 3000, 1000),
    "house":   (5,  (90, 330, 730, 895),     [(255,255,255)], 3000, 1200),
    "walk":    (6,  (780, 440, 1230, 985),   [(255,255,255)], 3000, 1000),
    "carer":   (8,  (845, 395, 1115, 725),   [(255,255,255)], 3000, 800),
    "social":  (9,  (835, 395, 1095, 725),   [(255,255,255)], 3000, 800),
    "paper":   (11, (272, 236, 806, 1030),   [], 0, 1300),
    "screen":  (11, (1168, 238, 1684, 1030), [], 0, 1300),
    "tlk_l":   (15, (365, 330, 610, 535),    [(255,255,255),(224,245,255)], 3000, 600),
    "tlk_r":   (15, (1455, 330, 1710, 535),  [(255,255,255),(224,245,255)], 3000, 600),
    "p_old":   (16, (140, 335, 345, 560),    [(255,255,255)], 2000, 600),
    "p_carer": (16, (440, 335, 640, 560),    [(255,255,255)], 2000, 600),
    "p_bld":   (16, (150, 630, 330, 860),    [(255,255,255)], 2000, 600),
    "p_sw":    (16, (440, 630, 625, 860),    [(255,255,255)], 2000, 600),
    "ph1":     (17, (205, 248, 545, 992),    [], 0, 1100),
    "ph2":     (17, (580, 248, 920, 992),    [], 0, 1100),
    "ph3":     (17, (945, 248, 1290, 992),   [], 0, 1100),
    "pc":      (18, (172, 238, 1348, 1026),  [], 0, 1800),
}

# 잘라낸 영역 안에서 지울 사각형(2000 기준): 제목 글자 조각, 장식 꽃 등
ERASE = {
    "hero": [(1250, 370, 1390, 470), (1780, 695, 1850, 780)],
    "walk": [(780, 540, 835, 605), (1175, 465, 1230, 525)],
    # 페르소나 장: 상자에서 인물로 이어지는 점선 조각(아래쪽 두 줄)
    "carer": [(845, 693, 936, 736), (1084, 696, 1125, 736)],
    "social": [(835, 704, 918, 736), (835, 698, 909, 704), (1084, 698, 1105, 736)],
}

def page(p):
    return Image.open(SRC / f"p{p}-{p:02d}.png").convert("RGB")

for name, (p, box, bgs, min_area, max_dim) in SPEC.items():
    im = page(p)
    x0, y0, x1, y1 = [v * 2 for v in box]
    crop = im.crop((x0, y0, x1, y1))
    arr = np.asarray(crop).astype(np.int16)
    for ex0, ey0, ex1, ey1 in ERASE.get(name, []):
        arr[(ey0*2-y0):(ey1*2-y0), (ex0*2-x0):(ex1*2-x0)] = 255
    h, w, _ = arr.shape
    if bgs:
        # 배경 후보: 지정 배경색과의 채널 합 차이가 작은 픽셀
        near = np.zeros((h, w), bool)
        for bg in bgs:
            d = np.abs(arr - np.array(bg, np.int16)).sum(axis=2)
            near |= d <= 48
        # 테두리에서 연결된 배경만 제거 (그림 안쪽의 흰색은 유지)
        lab, n = ndimage.label(near)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        border.discard(0)
        bgmask = np.isin(lab, list(border))
        fg = ~bgmask
        # 작은 전경 성분(점선 조각, 꽃, 따옴표) 제거
        flab, fn = ndimage.label(fg, structure=np.ones((3, 3)))
        sizes = ndimage.sum(fg, flab, range(1, fn + 1))
        keep = np.zeros(fn + 1, bool)
        keep[1:] = sizes >= min_area
        fg = keep[flab]
        alpha = (fg * 255).astype(np.uint8)
        # 가장자리 1px: 흰색과 섞인 정도만큼 반투명 (흰 테두리 방지)
        edge = fg & ~ndimage.binary_erosion(fg, iterations=1)
        lum = arr.min(axis=2)  # 가장 어두운 채널
        edge_alpha = np.clip((255 - lum) * 1.6, 0, 255).astype(np.uint8)
        alpha = np.where(edge, np.maximum(edge_alpha, 40), alpha).astype(np.uint8)
        rgba = np.dstack([arr.astype(np.uint8), alpha])
        out = Image.fromarray(rgba, "RGBA")
        # 투명 여백 잘라내기
        bbox = Image.fromarray(alpha).getbbox()
        if bbox:
            out = out.crop(bbox)
    else:
        out = crop
    if max(out.size) > max_dim:
        s = max_dim / max(out.size)
        out = out.resize((round(out.width * s), round(out.height * s)), Image.LANCZOS)
    out.save(OUT / f"{name}.webp", "WEBP", quality=90, method=6)
    print(f"{name:8s} page {p:2d} -> {out.size[0]}x{out.size[1]} {out.mode} {(OUT / (name + '.webp')).stat().st_size // 1024} KB")
