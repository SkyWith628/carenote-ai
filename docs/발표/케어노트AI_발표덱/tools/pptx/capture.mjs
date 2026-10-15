// HTML 덱 → 애니메이션 단위별 투명 PNG 레이어 + manifest.json
//
// 원리
//  · 덱의 등장 애니메이션 단위(.a, .draw, 도넛 circle.seg, 형광펜 .mark, 파이프라인 .pulse, 헤더 제목·탭)를
//    각각 "그 요소만 보이게" 해서 투명 배경으로 캡처한다. 자식 단위는 부모 캡처에서 숨겨 별도 레이어로 뺀다.
//  · 애니메이션이 하나도 없는 정적 부분(분홍 배경·카드·헤더)은 슬라이드마다 배경 이미지 1장으로 캡처한다.
//  · 각 단위의 효과 종류·지연·길이는 HTML 의 클래스와 CSS 변수(--base, --i, --step, --dur, --md)에서 그대로 읽는다.
//
// 사용법: node tools/pptx/capture.mjs index.html <출력폴더>
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch { pw = require('/opt/node-tools/node_modules/playwright'); }

const [, , htmlArg, outArg] = process.argv;
if (!htmlArg || !outArg) { console.error('usage: node capture.mjs index.html outDir'); process.exit(1); }
const out = path.resolve(outArg);
fs.mkdirSync(out, { recursive: true });
const url = pathToFileURL(path.resolve(htmlArg)).href;

const W = 1600, H = 900, SCALE = 2, PAD = 18;

// 캡처 전용 CSS: 조작 막대 숨김, 카드 세로 가운데 정렬, "지정한 요소만 보이기" 모드
const CAP_CSS = `
#nav,#hint,#bar,#notesPanel,#overview{display:none!important}
#card{top:46px!important}
body.cap,body.cap #viewport{background:transparent!important}
body.cap #stage *{visibility:hidden!important}
body.cap #stage [data-cap],body.cap #stage [data-cap] *{visibility:visible!important}
body.cap #stage marker,body.cap #stage marker *{visibility:visible!important}
body.cap #stage [data-cap] [data-cap-off],body.cap #stage [data-cap] [data-cap-off] *{visibility:hidden!important}
#stage [data-cap-off],#stage [data-cap-off] *{visibility:hidden!important}
body.cap:not(.capmark) .mark{background:none!important}
body.cap.capmark [data-cap].mark{color:transparent!important;background-size:100% 100%!important}
`;

const browser = await pw.chromium.launch();
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));

await page.goto(url, { waitUntil: 'load' });
const info = await page.evaluate(() => [...document.querySelectorAll('.slide')].map(s => ({
  bare: s.classList.contains('bare'), sec: +s.dataset.sec, title: s.dataset.title || '', sub: s.dataset.sub || '',
})));
const firstOfSec = sec => info.findIndex(s => s.sec === sec);

const manifest = { width: W, height: H, scale: SCALE, slides: [] };

for (let n = 0; n < info.length; n++) {
  const cur = info[n], prev = n > 0 ? info[n - 1] : null;
  // 헤더(장 번호 탭·제목)는 HTML 처럼 바뀔 때만 움직인다
  let header = 'none', tab = false;
  if (!cur.bare) {
    if (!prev || prev.bare || prev.title !== cur.title) header = 'title';
    else if (prev.sub !== cur.sub) header = 'sub';
    tab = !prev || prev.bare || prev.sec !== cur.sec;
  }

  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto(url + '#s' + (n + 1));
  await page.reload({ waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: CAP_CSS });

  // ---- 1) 애니메이션 단위 수집 + 메타데이터 ----
  const meta = await page.evaluate(({ header, tab }) => {
    const ms = v => { v = (v || '').trim(); if (!v) return null; if (v.endsWith('ms')) return parseFloat(v); if (v.endsWith('s')) return parseFloat(v) * 1000; const f = parseFloat(v); return isNaN(f) ? null : f; };
    const slide = document.querySelector('.slide.active');
    const list = [...slide.querySelectorAll('.a, .draw, circle.seg, .mark, .pulse')];
    // 형광펜(.mark)은 글자 아래에 깔려야 하므로 자신을 품은 단위 바로 앞으로 옮긴다
    const ordered = [];
    for (const el of list) {
      if (el.classList.contains('mark')) continue;
      ordered.push(el);
    }
    for (const mk of list.filter(e => e.classList.contains('mark'))) {
      const host = mk.parentElement.closest('.a');
      const at = host ? ordered.indexOf(host) : -1;
      if (at >= 0) ordered.splice(at, 0, mk); else ordered.push(mk);
    }
    const hdr = [];
    if (tab) hdr.push({ el: document.getElementById('tab'), kind: 'tab' });
    if (header === 'title') hdr.push({ el: document.querySelector('#hdrTitle .t:last-child'), kind: 'title' });
    if (header === 'sub') hdr.push({ el: document.querySelector('#hdrTitle .t:last-child i'), kind: 'sub' });

    const all = ordered.map(el => ({ el, kind: 'unit' })).concat(hdr.filter(h => h.el));
    all.forEach((u, k) => u.el.setAttribute('data-uid', String(k)));

    const units = all.map((u, k) => {
      const el = u.el, cs = getComputedStyle(el), v = n => cs.getPropertyValue(n);
      const base = ms(v('--base')) ?? 180, step = ms(v('--step')) ?? 85, i = parseFloat(v('--i')) || 0;
      let delay = base + i * step, dur = 700, effect = 'up', loop = null;
      const has = c => el.classList.contains(c);
      if (u.kind === 'tab') { effect = 'flip'; delay = 0; dur = 600; }
      else if (u.kind === 'title') { effect = 'up'; delay = 0; dur = 600; }
      else if (u.kind === 'sub') { effect = 'right'; delay = 0; dur = 500; }
      else if (has('mark')) { effect = 'wipe-left'; delay = ms(v('--md')) ?? 1600; dur = 1000; }
      else if (el.matches('circle.seg')) {
        const host = el.closest('.a'); let pd = 0;
        if (host) { const hs = getComputedStyle(host); pd = (ms(hs.getPropertyValue('--base')) ?? 180) + (parseFloat(hs.getPropertyValue('--i')) || 0) * (ms(hs.getPropertyValue('--step')) ?? 85); }
        effect = 'wheel'; delay = Math.max(520, pd + 300); dur = 1200;
      } else if (has('pulse')) { effect = 'fade'; delay = 2600; dur = 300; loop = 'pulse'; }
      else if (has('draw')) {
        dur = ms(v('--dur')) ?? 900;
        if (el.closest('svg.ring')) effect = 'wheel';
        else {
          const L = el.getTotalLength(), m = el.getScreenCTM();
          const a = new DOMPoint(el.getPointAtLength(0).x, el.getPointAtLength(0).y).matrixTransform(m);
          const b = new DOMPoint(el.getPointAtLength(L).x, el.getPointAtLength(L).y).matrixTransform(m);
          const dx = b.x - a.x, dy = b.y - a.y;
          effect = Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? 'wipe-left' : 'wipe-right') : (dy >= 0 ? 'wipe-top' : 'wipe-bottom');
        }
      } else {
        effect = has('l') ? 'left' : has('r') ? 'right' : has('pop') ? 'pop' : has('fade') ? 'fade'
          : has('dn') ? 'down' : has('grow') ? 'wipe-bottom' : has('wide') ? 'wipe-left' : 'up';
      }
      if (has('star')) loop = 'twinkle';
      if (has('float')) loop = 'bob';
      // 이 단위 안에 든 다른 단위(형광펜 제외)는 캡처에서 숨겨 별도 레이어로
      const offs = all.filter((o, j) => j !== k && el.contains(o.el) && !o.el.classList.contains('mark')).map(o => +o.el.getAttribute('data-uid'));
      const go = el.dataset.go ? +el.dataset.go : null;
      const alt = (el.getAttribute('alt') || el.querySelector?.('img')?.getAttribute('alt') || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 180);
      return { uid: k, kind: u.kind, effect, delay: Math.round(delay), dur: Math.round(dur), loop, offs, go, alt, mark: has('mark'), pulse: has('pulse') };
    });
    const notes = (slide.querySelector('.notes')?.textContent || '').trim();
    return { units, notes };
  }, { header, tab });

  // HTML 타이머(숫자 카운트업·도넛 채우기)가 끝날 때까지 기다린 뒤, 모션을 끄고 최종 상태로 캡처
  await page.waitForTimeout(4500);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.evaluate(() => { const p = document.querySelector('.slide.active .pulse'); if (p) p.setAttribute('style', 'display:block;offset-distance:0%;opacity:1;animation:none'); });
  await page.waitForTimeout(250);

  const tag = String(n + 1).padStart(2, '0');
  const slideOut = { base: `s${tag}_base.jpg`, notes: meta.notes, units: [] };

  for (const u of meta.units) {
    const r = await page.evaluate(({ uid, offs, mark }) => {
      document.body.classList.add('cap');
      document.body.classList.toggle('capmark', !!mark);
      document.querySelectorAll('[data-cap],[data-cap-off]').forEach(e => { e.removeAttribute('data-cap'); e.removeAttribute('data-cap-off'); });
      const el = document.querySelector(`[data-uid="${uid}"]`);
      el.setAttribute('data-cap', '');
      offs.forEach(o => document.querySelector(`[data-uid="${o}"]`)?.setAttribute('data-cap-off', ''));
      const b = el.getBoundingClientRect();
      return { x: b.left, y: b.top, w: b.width, h: b.height };
    }, u);
    if (r.w < 1 && r.h < 1) continue; // 곧은 선은 폭이나 높이가 0 이므로 둘 다 0일 때만 건너뜀
    const x0 = Math.max(0, Math.floor(r.x - PAD)), y0 = Math.max(0, Math.floor(r.y - PAD));
    const x1 = Math.min(W, Math.ceil(r.x + r.w + PAD)), y1 = Math.min(H, Math.ceil(r.y + r.h + PAD));
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;
    const file = `s${tag}_u${String(u.uid).padStart(3, '0')}.png`;
    await page.screenshot({ path: path.join(out, file), clip: { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, omitBackground: true });
    slideOut.units.push({
      file, x: x0, y: y0, w: x1 - x0, h: y1 - y0, effect: u.effect, delay: u.delay, dur: u.dur, loop: u.loop,
      link: u.go != null ? firstOfSec(u.go) + 1 : null, alt: u.alt || `slide ${n + 1} element ${u.uid}`,
    });
  }

  // 배경(정적 부분) 1장: 모든 애니메이션 단위를 숨기고 캡처
  await page.evaluate(() => {
    document.body.classList.remove('cap', 'capmark');
    document.querySelectorAll('[data-cap]').forEach(e => e.removeAttribute('data-cap'));
    document.querySelectorAll('[data-uid]').forEach(e => e.setAttribute('data-cap-off', ''));
  });
  await page.screenshot({ path: path.join(out, slideOut.base), type: 'jpeg', quality: 95 });
  manifest.slides.push(slideOut);
  console.log(`slide ${n + 1}: ${slideOut.units.length} layers`);
}

fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 1));
await browser.close();
