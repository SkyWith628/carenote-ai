// manifest.json → PPTX (레이어 이미지를 HTML 과 같은 위치에 배치, 발표 노트 포함)
// 애니메이션은 다음 단계(animate.py)에서 PowerPoint 기본 애니메이션 XML 로 넣는다.
// 사용법: node tools/pptx/build.cjs <캡처폴더> <출력.pptx>
const path = require('path');
let pptxgen;
try { pptxgen = require('pptxgenjs'); } catch { pptxgen = require(path.resolve(process.env.PPTXGENJS_DIR || '.', 'node_modules/pptxgenjs')); }
const fs = require('fs');

const [, , capDir, outFile] = process.argv;
const m = JSON.parse(fs.readFileSync(path.join(capDir, 'manifest.json'), 'utf8'));
const K = 1 / 120; // LAYOUT_WIDE 13.333in × 7.5in = 1600 × 900 px → 1px = 1/120in

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = '케어노트 AI — 음성메모를 기록으로';
pres.subject = '한국경제신문 지능정보 SW 아이디어 경진대회';
pres.author = '고려대학교 지능정보 SW 아카데미 8기 6조';

m.slides.forEach((s, si) => {
  const slide = pres.addSlide();
  slide.background = { path: path.join(capDir, s.base) };
  s.units.forEach((u, k) => {
    const opt = {
      path: path.join(capDir, u.file),
      x: u.x * K, y: u.y * K, w: u.w * K, h: u.h * K,
      objectName: `u${k}`, altText: u.alt,
    };
    if (u.link) opt.hyperlink = { slide: u.link, tooltip: `${u.link}번 슬라이드로 이동` };
    slide.addImage(opt);
  });
  if (s.notes) slide.addNotes(s.notes);
});

pres.writeFile({ fileName: outFile }).then(f => console.log('wrote', f));
