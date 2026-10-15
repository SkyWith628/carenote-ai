#!/usr/bin/env python3
"""build.cjs 가 만든 PPTX 에 PowerPoint 기본 애니메이션(<p:timing>)과 화면 전환을 넣는다.

- 슬라이드가 열리면 모든 요소가 HTML 과 같은 지연·길이로 자동 등장한다(클릭 불필요, '이전 효과와 함께').
- 효과 대응
    up/down/left/right  → 나타내기 + 살짝 이동(떠오르기·날아오기)
    pop                 → 확대(0.6배 → 1.05배 → 1배) + 나타내기
    fade                → 나타내기
    wipe-*              → 닦아내기(선 그리기·막대 자라기·형광펜)
    wheel               → 시계 방향 원형(도넛 채우기·빨간 동그라미)
    flip                → 세로로 펼치기(장 번호 탭)
  반복 효과: twinkle(별 회전) · bob(위아래 둥실) · pulse(파이프라인 점 이동)
- 슬라이드 전환: 페이드

사용법: python3 tools/pptx/animate.py <캡처폴더/manifest.json> <입력.pptx> <출력.pptx>
"""
import json, re, sys, zipfile

manifest_path, src, dst = sys.argv[1:4]
m = json.load(open(manifest_path, encoding="utf-8"))
W, H = m["width"], m["height"]


class Ids:
    def __init__(self, start):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


def tgt(spid):
    return f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'


def set_visible(nid, spid):
    return (f'<p:set><p:cBhvr><p:cTn id="{nid()}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
            f'{tgt(spid)}<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
            f'<p:to><p:strVal val="visible"/></p:to></p:set>')


def anim_effect(nid, spid, dur, flt):
    return (f'<p:animEffect transition="in" filter="{flt}"><p:cBhvr><p:cTn id="{nid()}" dur="{dur}"/>{tgt(spid)}</p:cBhvr></p:animEffect>')


def anim_prop(nid, spid, dur, attr, tavs, ease=True):
    e = ' decel="100000"' if ease else ''
    tl = ''.join(f'<p:tav tm="{tm}"><p:val><p:strVal val="{val}"/></p:val></p:tav>' for tm, val in tavs)
    return (f'<p:anim calcmode="lin" valueType="num"><p:cBhvr><p:cTn id="{nid()}" dur="{dur}"{e} fill="hold"/>{tgt(spid)}'
            f'<p:attrNameLst><p:attrName>{attr}</p:attrName></p:attrNameLst></p:cBhvr><p:tavLst>{tl}</p:tavLst></p:anim>')


# 픽셀 이동량을 슬라이드 비율로 (HTML: 위아래 28px, 좌우 46px, 아래로 24px)
UP, SIDE, DOWN = 28 / H, 46 / W, 24 / H
# 닦아내기: PowerPoint 는 filter="wipe(방향)" 의 방향 쪽에서 시작한다
WIPE = {"wipe-left": ("wipe(left)", 8), "wipe-right": ("wipe(right)", 2),
        "wipe-top": ("wipe(up)", 1), "wipe-bottom": ("wipe(down)", 4)}


def entrance(nid, spid, u):
    eff, d, t = u["effect"], u["delay"], u["dur"]
    body = set_visible(nid, spid)
    pid, sub = 10, 0
    if eff in ("up", "down", "left", "right"):
        attr, start = {"up": ("ppt_y", f"#ppt_y+{UP:.4f}"), "down": ("ppt_y", f"#ppt_y-{DOWN:.4f}"),
                       "left": ("ppt_x", f"#ppt_x-{SIDE:.4f}"), "right": ("ppt_x", f"#ppt_x+{SIDE:.4f}")}[eff]
        body += anim_effect(nid, spid, t, "fade") + anim_prop(nid, spid, t, attr, [(0, start), (100000, f"#{attr}")])
        pid, sub = {"up": (42, 0), "down": (2, 1), "left": (2, 8), "right": (2, 2)}[eff]
    elif eff == "pop":
        body += anim_effect(nid, spid, int(t * 0.6), "fade")
        for a in ("ppt_w", "ppt_h"):
            body += anim_prop(nid, spid, t, a, [(0, f"#{a}*0.6"), (70000, f"#{a}*1.05"), (100000, f"#{a}")], ease=False)
        pid, sub = 53, 16
    elif eff == "fade":
        body += anim_effect(nid, spid, t, "fade")
    elif eff in WIPE:
        flt, sub = WIPE[eff]
        body += anim_effect(nid, spid, t, flt)
        pid = 22
    elif eff == "wheel":
        body += anim_effect(nid, spid, t, "wheel(1)")
        pid, sub = 21, 1
    elif eff == "flip":
        body += anim_effect(nid, spid, int(t * 0.5), "fade") + anim_prop(nid, spid, t, "ppt_h", [(0, "0"), (100000, "#ppt_h")])
        pid, sub = 53, 16
    return (f'<p:par><p:cTn id="{nid()}" presetID="{pid}" presetClass="entr" presetSubtype="{sub}" fill="hold" nodeType="withEffect">'
            f'<p:stCondLst><p:cond delay="{d}"/></p:stCondLst><p:childTnLst>{body}</p:childTnLst></p:cTn></p:par>')


def loop(nid, spid, u):
    kind, start = u["loop"], u["delay"] + u["dur"]
    if kind == "twinkle":   # 30° 회전하며 살짝 커졌다 돌아오기, 무한 반복
        inner = (f'<p:animRot by="1800000"><p:cBhvr><p:cTn id="{nid()}" dur="1500" autoRev="1" accel="50000" decel="50000" fill="hold"/>'
                 f'{tgt(spid)}<p:attrNameLst><p:attrName>r</p:attrName></p:attrNameLst></p:cBhvr></p:animRot>'
                 f'<p:animScale><p:cBhvr><p:cTn id="{nid()}" dur="1500" autoRev="1" accel="50000" decel="50000" fill="hold"/>{tgt(spid)}</p:cBhvr>'
                 f'<p:by x="115000" y="115000"/></p:animScale>')
        pid, cls = 8, "emph"
    elif kind == "bob":     # 10px 위아래 둥실
        inner = (f'<p:animMotion origin="layout" path="M 0 0 L 0 -{10 / H:.4f} E" pathEditMode="relative" ptsTypes="AA">'
                 f'<p:cBhvr><p:cTn id="{nid()}" dur="2000" autoRev="1" accel="50000" decel="50000" fill="hold"/>{tgt(spid)}'
                 f'<p:attrNameLst><p:attrName>ppt_x</p:attrName><p:attrName>ppt_y</p:attrName></p:attrNameLst></p:cBhvr>'
                 f'<p:rCtr x="0" y="0"/></p:animMotion>')
        pid, cls = 0, "path"
    elif kind == "pulse":   # 파이프라인을 따라 도는 점 (HTML offset-path 와 같은 사각 경로)
        dx, dy = 760 / W, 320 / H
        inner = (f'<p:animMotion origin="layout" path="M 0 0 L {dx:.4f} 0 L {dx:.4f} {dy:.4f} L 0 {dy:.4f} E" pathEditMode="relative" ptsTypes="AAAA">'
                 f'<p:cBhvr><p:cTn id="{nid()}" dur="6000" fill="hold"/>{tgt(spid)}'
                 f'<p:attrNameLst><p:attrName>ppt_x</p:attrName><p:attrName>ppt_y</p:attrName></p:attrNameLst></p:cBhvr>'
                 f'<p:rCtr x="0" y="0"/></p:animMotion>')
        pid, cls = 0, "path"
    else:
        return ""
    return (f'<p:par><p:cTn id="{nid()}" presetID="{pid}" presetClass="{cls}" presetSubtype="0" repeatCount="indefinite" fill="hold" nodeType="withEffect">'
            f'<p:stCondLst><p:cond delay="{start}"/></p:stCondLst><p:childTnLst>{inner}</p:childTnLst></p:cTn></p:par>')


def timing_xml(units, spids):
    nid = Ids(4)
    effects = ""
    for k, u in enumerate(units):
        if k not in spids:
            continue
        effects += entrance(nid, spids[k], u)
        if u.get("loop"):
            effects += loop(nid, spids[k], u)
    if not effects:
        return ""
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            '<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
            '<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'{effects}'
            '</p:childTnLst></p:cTn></p:par>'
            '</p:childTnLst></p:cTn></p:par>'
            '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
            '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
            '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')


TRANSITION = '<p:transition spd="med"><p:fade/></p:transition>'

zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
total = 0
for item in zin.infolist():
    data = zin.read(item.filename)
    mt = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", item.filename)
    if mt:
        idx = int(mt.group(1)) - 1
        xml = data.decode("utf-8")
        spids = {int(n): sid for sid, n in re.findall(r'<p:cNvPr id="(\d+)" name="u(\d+)"', xml)}
        units = m["slides"][idx]["units"]
        tim = timing_xml(units, spids)
        total += len(spids)
        add = TRANSITION + tim
        if "<p:extLst>" in xml.split("</p:cSld>", 1)[1]:
            head, tail = xml.split("</p:cSld>", 1)
            tail = tail.replace("<p:extLst>", add + "<p:extLst>", 1)
            xml = head + "</p:cSld>" + tail
        else:
            xml = xml.replace("</p:sld>", add + "</p:sld>")
        data = xml.encode("utf-8")
    zout.writestr(item, data)
zout.close()
print(f"animated {total} layers across {len(m['slides'])} slides -> {dst}")
