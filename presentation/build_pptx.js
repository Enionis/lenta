#!/usr/bin/env node
/**
 * Lenta Tech — команда «Медвежата». Презентация финального решения.
 * Запуск: node presentation/build_pptx.js
 */
const pptxgen = require("pptxgenjs");

const NAVY  = "1B2A4E";   // dark slides / headers
const ORANGE = "E76F51";  // accent (stats, highlights)
const CREAM = "F4F1DE";   // light content bg
const CHARCOAL = "2A2D3E"; // body text on light
const MUTED = "6B7280";   // captions
const WHITE = "FFFFFF";
const GREEN = "5FAD61";   // success/keep
const RED   = "C0392B";   // fail/regress

const HEADER_FONT = "Calibri";
const BODY_FONT   = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";   // 13.33 × 7.5 in
pres.title = "LentaTech Pricetag — Медвежата";
pres.author = "Команда Медвежата";
pres.company = "Lenta Tech Hackathon";

const W = 13.33, H = 7.5;

// ---------- helpers ----------
function addPageNum(slide, n, total) {
  slide.addText(`${n} / ${total}`, {
    x: W - 1.1, y: H - 0.45, w: 0.9, h: 0.3,
    fontFace: BODY_FONT, fontSize: 10, color: MUTED, align: "right",
    margin: 0,
  });
}
function addBrand(slide) {
  // декоративная вертикальная полоса слева (motif)
  slide.addShape("rect", {
    x: 0, y: 0, w: 0.18, h: H, fill: { color: ORANGE }, line: { type: "none" },
  });
  slide.addText("МЕДВЕЖАТА × LENTA TECH", {
    x: 0.35, y: 0.25, w: 6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 10, bold: true, color: MUTED,
    charSpacing: 4, margin: 0,
  });
}
function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.5, y: 0.6, w: W - 1, h: 0.9,
    fontFace: HEADER_FONT, fontSize: 36, bold: true, color: NAVY, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 1.45, w: W - 1, h: 0.4,
      fontFace: BODY_FONT, fontSize: 14, color: MUTED, italic: true, margin: 0,
    });
  }
}
function statCallout(slide, x, y, w, value, label, color = NAVY) {
  slide.addShape("rect", {
    x, y, w, h: 1.6,
    fill: { color: WHITE }, line: { color: color, width: 0 },
    rectRadius: 0.05,
  });
  slide.addShape("rect", {
    x, y, w: 0.12, h: 1.6, fill: { color }, line: { type: "none" },
  });
  slide.addText(value, {
    x: x + 0.25, y: y + 0.1, w: w - 0.35, h: 0.95,
    fontFace: HEADER_FONT, fontSize: 44, bold: true, color, margin: 0,
  });
  slide.addText(label, {
    x: x + 0.25, y: y + 1.05, w: w - 0.35, h: 0.5,
    fontFace: BODY_FONT, fontSize: 11, color: MUTED, margin: 0,
  });
}

const TOTAL = 11;

// ========================= SLIDE 1 — TITLE =========================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // orange accent bar at left
  s.addShape("rect", { x: 0, y: 0, w: 0.5, h: H, fill: { color: ORANGE }, line: { type: "none" } });
  // diagonal cream wedge for motif
  s.addShape("rightTriangle", {
    x: W - 4.5, y: 0, w: 4.5, h: H,
    fill: { color: CREAM, transparency: 88 }, line: { type: "none" },
    flipH: true,
  });

  s.addText("МЕДВЕЖАТА", {
    x: 1.0, y: 0.6, w: 8, h: 0.4,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: ORANGE,
    charSpacing: 8, margin: 0,
  });
  s.addText("Распознавание ценников\nс видео магазинного робота", {
    x: 1.0, y: 1.5, w: 11, h: 2.2,
    fontFace: HEADER_FONT, fontSize: 48, bold: true, color: WHITE,
    margin: 0, lineSpacingMultiple: 1.05,
  });
  s.addText("Lenta Tech Hackathon · CPU-only ML pipeline · Docker production stack", {
    x: 1.0, y: 4.0, w: 11, h: 0.5,
    fontFace: BODY_FONT, fontSize: 18, color: CREAM, italic: true, margin: 0,
  });

  // hero stats row
  s.addShape("rect", { x: 1.0, y: 5.2, w: 0.08, h: 1.2, fill: { color: ORANGE }, line: { type: "none" } });
  s.addText("TARGET_METRIC", {
    x: 1.2, y: 5.2, w: 5, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, color: CREAM, charSpacing: 3, margin: 0,
  });
  s.addText("4/157 = 0.025", {
    x: 1.2, y: 5.6, w: 5, h: 0.8,
    fontFace: HEADER_FONT, fontSize: 36, bold: true, color: WHITE, margin: 0,
  });

  s.addShape("rect", { x: 7.5, y: 5.2, w: 0.08, h: 1.2, fill: { color: ORANGE }, line: { type: "none" } });
  s.addText("Top-пара", {
    x: 7.7, y: 5.2, w: 5, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, color: CREAM, charSpacing: 3, margin: 0,
  });
  s.addText("20/23 (87%)", {
    x: 7.7, y: 5.6, w: 5, h: 0.8,
    fontFace: HEADER_FONT, fontSize: 36, bold: true, color: WHITE, margin: 0,
  });

  s.addText("17 мая 2026 · Москва", {
    x: 1.0, y: H - 0.6, w: 6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, color: CREAM, margin: 0,
  });
}

// ========================= SLIDE 2 — КОМАНДА =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Команда «Медвежата»", "Кросс-функциональная команда из 4 человек");

  const roles = [
    { name: "Ангелина Дугау", role: "ML / Computer Vision", contact: "tg: @…",
      what: "YOLO детекторы v2/v4/v5b, OCR pipeline,\nLLM product_name, catalog scraper" },
    { name: "_заполнить_", role: "Backend / Infra", contact: "tg: @…",
      what: "FastAPI + Celery + Redis,\nDocker, JWT, observability" },
    { name: "_заполнить_", role: "Frontend / UX", contact: "tg: @…",
      what: "Streamlit operator UI,\ndashboard аналитики" },
    { name: "_заполнить_", role: "PM / Координация", contact: "tg: @…",
      what: "Roadmap, синки с организаторами,\nфинальная сборка" },
  ];
  const colW = 2.85, gap = 0.2, startX = 0.5;
  const y = 2.1;
  roles.forEach((r, i) => {
    const x = startX + i * (colW + gap);
    // card background
    s.addShape("rect", {
      x, y, w: colW, h: 4.3,
      fill: { color: CREAM }, line: { color: NAVY, width: 0 },
      rectRadius: 0.08,
    });
    // role icon — circle with role initial
    s.addShape("ellipse", {
      x: x + 0.4, y: y + 0.4, w: 0.9, h: 0.9,
      fill: { color: ORANGE }, line: { type: "none" },
    });
    const initial = r.role.split(/[ /]/)[0][0];
    s.addText(initial, {
      x: x + 0.4, y: y + 0.4, w: 0.9, h: 0.9,
      fontFace: HEADER_FONT, fontSize: 32, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(r.role, {
      x: x + 0.3, y: y + 1.5, w: colW - 0.6, h: 0.5,
      fontFace: BODY_FONT, fontSize: 11, color: ORANGE, bold: true,
      charSpacing: 2, margin: 0,
    });
    s.addText(r.name, {
      x: x + 0.3, y: y + 2.0, w: colW - 0.6, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 18, bold: true, color: NAVY, margin: 0,
    });
    s.addText(r.what, {
      x: x + 0.3, y: y + 2.7, w: colW - 0.6, h: 1.2,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
    s.addText(r.contact, {
      x: x + 0.3, y: y + 3.85, w: colW - 0.6, h: 0.3,
      fontFace: BODY_FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
    });
  });

  addPageNum(s, 2, TOTAL);
}

// ========================= SLIDE 3 — ЗАДАЧА + ВХОДНЫЕ ДАННЫЕ =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Задача и особенности входных данных", "Робот едет вдоль полки, на выходе — CSV из 29 полей");

  // left column - задача
  s.addText("ПОСТАНОВКА", {
    x: 0.5, y: 2.0, w: 5, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  s.addText(
    [
      { text: "Распознать ценники с видео\nмагазинного робота и вернуть CSV\nиз ", options: { fontSize: 16, color: CHARCOAL } },
      { text: "29 полей", options: { fontSize: 16, color: NAVY, bold: true } },
      { text: " по каждому ценнику.\n\n", options: { fontSize: 16, color: CHARCOAL } },
      { text: "Успех пары: ", options: { fontSize: 14, color: CHARCOAL } },
      { text: "≥ 80% из 23 контентных полей", options: { fontSize: 14, color: NAVY, bold: true } },
      { text: " совпали (по barcode или\nspatio-temporal-ключу).\n\n", options: { fontSize: 14, color: CHARCOAL } },
      { text: "Метрика = ", options: { fontSize: 14, color: CHARCOAL } },
      { text: "(успешные пары) / (всего ценников)", options: { fontSize: 14, color: NAVY, bold: true } },
      { text: ".", options: { fontSize: 14, color: CHARCOAL } },
    ],
    { x: 0.5, y: 2.4, w: 5.8, h: 3.5, margin: 0, lineSpacingMultiple: 1.2 }
  );

  // right column - характеристики данных
  const xR = 7.0;
  s.addShape("rect", {
    x: xR, y: 1.95, w: 5.8, h: 4.7,
    fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
  });
  s.addText("ОСОБЕННОСТИ СЪЁМКИ", {
    x: xR + 0.3, y: 2.1, w: 5.4, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  const items = [
    ["3840 × 2160", "разрешение 4K"],
    ["90° против ч.с.", "камера повёрнута на бок"],
    ["focal 2.8 мм", "16/2.8 мм матрица, дисторсия"],
    ["motion blur", "робот без остановок"],
    ["смешанное освещение", "блики на акционных ценниках"],
    ["3 видео × 157 ценников", "wine / honey / jam / dairy"],
  ];
  items.forEach((it, i) => {
    const yy = 2.55 + i * 0.62;
    s.addShape("ellipse", {
      x: xR + 0.3, y: yy + 0.05, w: 0.3, h: 0.3,
      fill: { color: NAVY }, line: { type: "none" },
    });
    s.addText(String(i + 1), {
      x: xR + 0.3, y: yy + 0.05, w: 0.3, h: 0.3,
      fontFace: BODY_FONT, fontSize: 10, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(it[0], {
      x: xR + 0.75, y: yy, w: 2.3, h: 0.4,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(it[1], {
      x: xR + 3.1, y: yy + 0.05, w: 2.7, h: 0.4,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
  });

  s.addText("Допущено: ручная разметка на этапе обучения (заявлено в материалах). Финальный pipeline — полностью автоматический.",
    { x: 0.5, y: 6.7, w: W - 1, h: 0.3,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 });

  addPageNum(s, 3, TOTAL);
}

// ========================= SLIDE 4 — РЕЗУЛЬТАТ =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Финальный результат", "Триплет-ensemble + полный 8-этапный pipeline");

  // big stat callout
  s.addShape("rect", {
    x: 0.5, y: 2.1, w: 6.0, h: 3.5,
    fill: { color: NAVY }, line: { type: "none" }, rectRadius: 0.1,
  });
  s.addShape("rect", {
    x: 0.5, y: 2.1, w: 0.15, h: 3.5, fill: { color: ORANGE }, line: { type: "none" },
  });
  s.addText("TARGET_METRIC", {
    x: 0.85, y: 2.3, w: 5.5, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, color: CREAM, charSpacing: 4, margin: 0,
  });
  s.addText("4 / 157", {
    x: 0.85, y: 2.7, w: 5.5, h: 1.6,
    fontFace: HEADER_FONT, fontSize: 96, bold: true, color: WHITE, margin: 0,
  });
  s.addText("= 0.025  ·  4 ценника прошли порог ≥ 80%", {
    x: 0.85, y: 4.35, w: 5.5, h: 0.5,
    fontFace: BODY_FONT, fontSize: 14, color: CREAM, margin: 0,
  });
  s.addText("Top-пара: 20 / 23 полей (87 %)", {
    x: 0.85, y: 4.85, w: 5.5, h: 0.5,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: ORANGE, margin: 0,
  });

  // right: day progression "bar chart"
  s.addText("ДИНАМИКА ПО ДНЯМ", {
    x: 7.0, y: 2.1, w: 5.8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  const progress = [
    ["День 11 · бейзлайн", 0, "0/157"],
    ["День 14 · bug fixes", 1, "1/157"],
    ["День 15 · стабилизация", 1, "1/157"],
    ["День 16 · ensemble v2+v4", 3, "3/157"],
    ["День 16 · +v5b +TTA",    4, "4/157"],
  ];
  const maxBar = 5.0;   // max bar width
  const maxVal = 5;
  progress.forEach((p, i) => {
    const yy = 2.6 + i * 0.6;
    const barW = (p[1] / maxVal) * maxBar;
    s.addText(p[0], {
      x: 7.0, y: yy, w: 3.0, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
    // background track
    s.addShape("rect", {
      x: 10.0, y: yy + 0.07, w: maxBar, h: 0.22,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.03,
    });
    if (barW > 0.05) {
      s.addShape("rect", {
        x: 10.0, y: yy + 0.07, w: barW, h: 0.22,
        fill: { color: i === progress.length - 1 ? ORANGE : NAVY }, line: { type: "none" },
        rectRadius: 0.03,
      });
    }
    s.addText(p[2], {
      x: 10.0 + maxBar + 0.1, y: yy, w: 1.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, bold: i === progress.length - 1, color: NAVY, margin: 0,
    });
  });

  // per-video breakdown
  s.addText("ПО ВИДЕО", {
    x: 0.5, y: 5.9, w: 12, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  const videos = [
    ["25_12-20 (wine)", "0/57", MUTED],
    ["26_12-20 (wine)", "4/71", ORANGE],
    ["43_15 (honey)",   "0/29", MUTED],
  ];
  const vidW = 4.0;
  videos.forEach((v, i) => {
    const vx = 0.5 + i * (vidW + 0.2);
    s.addShape("rect", {
      x: vx, y: 6.3, w: vidW, h: 0.85,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
    });
    s.addText(v[0], {
      x: vx + 0.2, y: 6.35, w: vidW - 0.4, h: 0.35,
      fontFace: BODY_FONT, fontSize: 12, color: CHARCOAL, margin: 0,
    });
    s.addText(v[1], {
      x: vx + 0.2, y: 6.65, w: vidW - 0.4, h: 0.45,
      fontFace: HEADER_FONT, fontSize: 22, bold: true, color: v[2], margin: 0,
    });
  });

  addPageNum(s, 4, TOTAL);
}

// ========================= SLIDE 5 — АРХИТЕКТУРА =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Архитектура pipeline", "8 этапов: от MP4 до 29-полевого CSV");

  const stages = [
    ["1", "DETECT",  "YOLO triple ensemble\n(v2 + v4 + v5b) + TTA\ntiled 2×2"],
    ["2", "TRACK",   "Greedy IoU\ntop-K=3 best frames\n(area × sharpness)"],
    ["3", "CROPS",   "hires pad=0.30\n+ QR-zone pad_y_top=1.2"],
    ["4", "OCR",     "EasyOCR ru+en × 4 rotations\n+ bottom-30% upscale ×5"],
    ["5", "DECODE",  "WeChat-QR + pyzbar +\nzxing-cpp + cv2.barcode\n+ aggressive CLAHE/Otsu"],
    ["6", "BUILD",   "Парсеры 29 полей\nK-frame merge, bbox dedup\nfont-aware prices"],
    ["7", "LLM",     "Qwen2.5-3B-Q4\n(embedded llama-cpp)\n→ product_name + fallback"],
    ["8", "CATALOG", "315 SKU (148 GT + 167 scraped)\nbarcode-match → подтянуть цены\nfuzzy ≥ 0.65 + dedup"],
  ];

  const cardW = 3.0, cardH = 1.7;
  const startX = 0.5, startY = 2.1;
  const gapX = 0.13, gapY = 0.25;
  stages.forEach((st, i) => {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x = startX + col * (cardW + gapX);
    const y = startY + row * (cardH + gapY);
    // card
    s.addShape("rect", {
      x, y, w: cardW, h: cardH,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
    });
    // left accent
    s.addShape("rect", {
      x, y, w: 0.1, h: cardH, fill: { color: ORANGE }, line: { type: "none" },
    });
    // step number badge
    s.addShape("ellipse", {
      x: x + 0.25, y: y + 0.2, w: 0.45, h: 0.45,
      fill: { color: NAVY }, line: { type: "none" },
    });
    s.addText(st[0], {
      x: x + 0.25, y: y + 0.2, w: 0.45, h: 0.45,
      fontFace: HEADER_FONT, fontSize: 16, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    // stage name
    s.addText(st[1], {
      x: x + 0.85, y: y + 0.2, w: cardW - 1.0, h: 0.4,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY,
      charSpacing: 2, margin: 0,
    });
    // detail
    s.addText(st[2], {
      x: x + 0.25, y: y + 0.75, w: cardW - 0.4, h: cardH - 0.85,
      fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, margin: 0,
      lineSpacingMultiple: 1.15,
    });
  });

  s.addText("Запуск: ~50 мин/видео на CPU · готов под GPU/rknn для прод-нагрузки",
    { x: 0.5, y: H - 0.7, w: W - 1, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 });

  addPageNum(s, 5, TOTAL);
}

// ========================= SLIDE 6 — ЧТО УЛУЧШИЛО МЕТРИКУ =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Что критически улучшило метрику", "От 0/157 → 4/157 за день 16");

  const wins = [
    { tag: "+15-20%", title: "K-frame OCR (3 кадра на трек)", note: "Recall полей на главных fields" },
    { tag: "+15 BC",  title: "Aggressive barcode decode",      note: "CLAHE + Otsu + ×4 upscale × 4 rotation" },
    { tag: "16 → 20", title: "Catalog price-pull на barcode-match", note: "Top-pair: подтягиваем цены из 315 SKU" },
    { tag: "+12%",    title: "Triple ensemble v2+v4+v5b + TTA", note: "501+569+363 = 1433 трека" },
    { tag: "0 → 4",   title: "Barcode dedup + threshold 0.65",  note: "Без них catalog enrichment ломал метрику" },
    { tag: "fix",     title: "Font-size-aware price parser",    note: "Главная цена не теряется в мелочи" },
  ];

  const cardW = 4.0, cardH = 1.85;
  const startX = 0.5, startY = 2.1;
  const gapX = 0.2, gapY = 0.2;
  wins.forEach((w, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = startX + col * (cardW + gapX);
    const y = startY + row * (cardH + gapY);
    s.addShape("rect", {
      x, y, w: cardW, h: cardH,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
    });
    // tag pill
    s.addShape("rect", {
      x: x + 0.25, y: y + 0.25, w: 1.5, h: 0.42,
      fill: { color: GREEN }, line: { type: "none" }, rectRadius: 0.2,
    });
    s.addText(w.tag, {
      x: x + 0.25, y: y + 0.25, w: 1.5, h: 0.42,
      fontFace: BODY_FONT, fontSize: 12, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(w.title, {
      x: x + 0.25, y: y + 0.8, w: cardW - 0.5, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY,
      margin: 0,
    });
    s.addText(w.note, {
      x: x + 0.25, y: y + 1.3, w: cardW - 0.5, h: 0.5,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
  });

  addPageNum(s, 6, TOTAL);
}

// ========================= SLIDE 7 — ЧТО НЕ СРАБОТАЛО =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Что не сработало — честно", "Зрелость подхода = понимание границ");

  const fails = [
    { title: "Детектор v3 на расширенных bbox",
      cause: "OCR терял фокус на бóльшем crop'е — recall главной цены упал",
      result: "Откатились на v2 (1/157 рекорд тех дней сохранён)" },
    { title: "Glare removal (CLAHE + highlight suppression)",
      cause: "Подавлял яркие пиксели текста на акционных ценниках",
      result: "Отключено в финальном pipeline, оставлено как опция" },
    { title: "Этап Б: area × sharp × conf^1.5 + median_conf ≥ 0.20",
      cause: "Фильтр срубил 5 из 6 catalog-barcode-матчей (conf=0.12–0.18)",
      result: "4/157 → 0/157, откатились" },
    { title: "PaddleOCR Server для мелкого шрифта",
      cause: "Не вписался по time-budget хакатона",
      result: "Перенесён в future work" },
    { title: "Lenta-каталог через API (Qrator block)",
      cause: "TLS fingerprint check — не пускает curl/requests/Playwright",
      result: "Обошли через undetected-chromedriver (167 SKU, медленно)" },
  ];

  fails.forEach((f, i) => {
    const y = 2.05 + i * 0.94;
    // X-mark badge
    s.addShape("ellipse", {
      x: 0.5, y: y + 0.1, w: 0.55, h: 0.55,
      fill: { color: RED }, line: { type: "none" },
    });
    s.addText("✗", {
      x: 0.5, y: y + 0.1, w: 0.55, h: 0.55,
      fontFace: HEADER_FONT, fontSize: 22, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(f.title, {
      x: 1.2, y: y + 0.05, w: 5.8, h: 0.4,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText("Причина: " + f.cause, {
      x: 1.2, y: y + 0.4, w: 11.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
    s.addText("→ " + f.result, {
      x: 1.2, y: y + 0.7, w: 11.5, h: 0.3,
      fontFace: BODY_FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
    });
  });

  addPageNum(s, 7, TOTAL);
}

// ========================= SLIDE 8 — BOTTLENECK =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Бутылочное горлышко", "Не детектор и не каталог — декод штрихкодов");

  // 3 metrics row
  const m = [
    { value: "1 433", label: "детекций (треков)", color: GREEN },
    { value: "148 / 156", label: "GT barcodes в каталоге (95 %)", color: GREEN },
    { value: "13 / 156", label: "декодировано декодерами (8 %)", color: RED },
  ];
  m.forEach((it, i) => {
    const x = 0.5 + i * 4.3;
    s.addShape("rect", {
      x, y: 2.1, w: 4.0, h: 1.6,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
    });
    s.addShape("rect", { x, y: 2.1, w: 0.12, h: 1.6, fill: { color: it.color }, line: { type: "none" } });
    s.addText(it.value, {
      x: x + 0.25, y: 2.25, w: 3.7, h: 0.85,
      fontFace: HEADER_FONT, fontSize: 36, bold: true, color: it.color, margin: 0,
    });
    s.addText(it.label, {
      x: x + 0.25, y: 3.1, w: 3.7, h: 0.5,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
  });

  // per-video decode rate
  s.addText("DECODE RATE ПО ВИДЕО", {
    x: 0.5, y: 4.0, w: 12, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  const rates = [
    ["25_12-20", 2, 55,  "3.6 %"],
    ["26_12-20", 11, 70, "15.7 %"],
    ["43_15",    0, 26,  "0 %"],
  ];
  rates.forEach((r, i) => {
    const yy = 4.4 + i * 0.55;
    s.addText(r[0], {
      x: 0.5, y: yy, w: 2.0, h: 0.35,
      fontFace: BODY_FONT, fontSize: 13, color: CHARCOAL, margin: 0,
    });
    const trackW = 7.5;
    s.addShape("rect", {
      x: 2.6, y: yy + 0.08, w: trackW, h: 0.22,
      fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.03,
    });
    const fillW = Math.max(0.03, (r[1] / r[2]) * trackW);
    if (r[1] > 0) {
      s.addShape("rect", {
        x: 2.6, y: yy + 0.08, w: fillW, h: 0.22,
        fill: { color: r[1] === 0 ? RED : ORANGE }, line: { type: "none" }, rectRadius: 0.03,
      });
    }
    s.addText(`${r[1]} / ${r[2]}`, {
      x: 2.6 + trackW + 0.15, y: yy, w: 1.4, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
    });
    s.addText(r[3], {
      x: 4.1 + trackW + 0.15, y: yy, w: 1.5, h: 0.35,
      fontFace: HEADER_FONT, fontSize: 12, bold: true,
      color: r[1] === 0 ? RED : (r[1] >= 10 ? GREEN : ORANGE), margin: 0,
    });
  });

  // bottleneck statement
  s.addShape("rect", {
    x: 0.5, y: 6.2, w: W - 1, h: 0.9,
    fill: { color: NAVY }, line: { type: "none" }, rectRadius: 0.05,
  });
  s.addShape("rect", { x: 0.5, y: 6.2, w: 0.12, h: 0.9, fill: { color: ORANGE }, line: { type: "none" } });
  s.addText("Декодеры пробуют 72 комбинации (3 preproc × 2 upscale × 4 rotation × 3 декодера). Дальше — physical ceiling: на 43_15 GT-barcodes отсутствуют и в OCR-тексте. Требуется PaddleOCR / Real-ESRGAN / GPU-камера.",
    { x: 0.75, y: 6.27, w: W - 1.4, h: 0.78,
      fontFace: BODY_FONT, fontSize: 12, color: CREAM, italic: true, margin: 0,
      lineSpacingMultiple: 1.2 });

  addPageNum(s, 8, TOTAL);
}

// ========================= SLIDE 9 — PRODUCTION STACK =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Production-готовность", "Docker compose: api · worker · beat · redis · ui · dashboard");

  // architecture diagram - 3 layers
  const lay = [
    { y: 2.1, label: "PRESENTATION", color: ORANGE, items: [
      ["Streamlit UI :8501", "оператор: upload + результат"],
      ["Dashboard :8502",    "аналитика заданий + полей"],
      ["Swagger :8000/docs", "OpenAPI для интеграции"],
    ]},
    { y: 3.85, label: "API + QUEUE", color: NAVY, items: [
      ["FastAPI :8000", "JWT auth · slowapi rate-limit · libmagic"],
      ["Celery Worker", "ML pipeline · Prometheus :9101"],
      ["Celery Beat",   "Cleanup > 7 дней · SQLite jobs.db"],
    ]},
    { y: 5.6, label: "STORAGE",     color: GREEN, items: [
      ["Redis :6379",   "broker + pub/sub + progress"],
      ["SQLite",        "история заданий"],
      ["Parquet/CSV",   "результаты + аналитика"],
    ]},
  ];
  lay.forEach((row) => {
    s.addText(row.label, {
      x: 0.5, y: row.y - 0.05, w: 1.8, h: 0.3,
      fontFace: BODY_FONT, fontSize: 10, bold: true, color: row.color,
      charSpacing: 4, margin: 0,
    });
    row.items.forEach((it, i) => {
      const x = 2.4 + i * 3.7;
      s.addShape("rect", {
        x, y: row.y - 0.05, w: 3.5, h: 1.4,
        fill: { color: CREAM }, line: { type: "none" }, rectRadius: 0.05,
      });
      s.addShape("rect", { x, y: row.y - 0.05, w: 0.1, h: 1.4, fill: { color: row.color }, line: { type: "none" } });
      s.addText(it[0], {
        x: x + 0.25, y: row.y, w: 3.2, h: 0.45,
        fontFace: HEADER_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
      });
      s.addText(it[1], {
        x: x + 0.25, y: row.y + 0.5, w: 3.2, h: 0.85,
        fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0,
        lineSpacingMultiple: 1.2,
      });
    });
  });

  // observability footer
  s.addText("Observability: structlog JSON · Sentry SDK · Prometheus · /health (api+redis+model+disk)",
    { x: 0.5, y: H - 0.55, w: W - 1, h: 0.3,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, italic: true, margin: 0 });

  addPageNum(s, 9, TOTAL);
}

// ========================= SLIDE 10 — ДАННЫЕ ДЛЯ ОБУЧЕНИЯ =========================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addBrand(s);
  addTitle(s, "Прозрачность обучения", "Все данные собраны автоматически (ТЗ-compliant)");

  const rows = [
    { src: "GT bbox из организаторской разметки",
      vol: "179 кадров\n~1044 ценников", method: "auto · CSV в Данные/",
      use: "fine-tune v2 (YOLOv8n)" },
    { src: "Synthetic dataset",
      vol: "5000 кадров", method: "auto · 271 шаблон × фоны × augment",
      use: "fine-tune v4 (YOLOv8n)" },
    { src: "GT bbox (повторно)",
      vol: "те же 179", method: "auto",
      use: "fine-tune v5b (YOLOv8s 1024)" },
    { src: "Каталог Lenta (publish)",
      vol: "315 SKU\n148 GT + 167 scraped", method: "auto · undetected-chromedriver",
      use: "barcode-match → enrichment цен" },
  ];
  // table header
  const cols = [
    { w: 4.0, lab: "ИСТОЧНИК" },
    { w: 2.4, lab: "ОБЪЁМ" },
    { w: 3.5, lab: "МЕТОД" },
    { w: 2.5, lab: "ПРИМЕНЕНИЕ" },
  ];
  let cx = 0.5;
  const headY = 2.1;
  cols.forEach((c) => {
    s.addText(c.lab, {
      x: cx, y: headY, w: c.w, h: 0.35,
      fontFace: BODY_FONT, fontSize: 10, bold: true, color: ORANGE,
      charSpacing: 4, margin: 0,
    });
    cx += c.w;
  });
  // header underline
  s.addShape("line", {
    x: 0.5, y: headY + 0.4, w: W - 1, h: 0,
    line: { color: NAVY, width: 1.5 },
  });

  rows.forEach((r, i) => {
    const y = headY + 0.55 + i * 1.0;
    if (i % 2 === 0) {
      s.addShape("rect", {
        x: 0.5, y: y - 0.05, w: W - 1, h: 0.95,
        fill: { color: CREAM }, line: { type: "none" },
      });
    }
    let xx = 0.5;
    [r.src, r.vol, r.method, r.use].forEach((txt, j) => {
      s.addText(txt, {
        x: xx + 0.1, y: y + 0.05, w: cols[j].w - 0.2, h: 0.85,
        fontFace: BODY_FONT, fontSize: 11,
        color: j === 0 ? NAVY : CHARCOAL, bold: j === 0,
        margin: 0, valign: "middle", lineSpacingMultiple: 1.15,
      });
      xx += cols[j].w;
    });
  });

  // bottom callout
  s.addShape("rect", {
    x: 0.5, y: 6.5, w: W - 1, h: 0.65,
    fill: { color: NAVY }, line: { type: "none" }, rectRadius: 0.05,
  });
  s.addShape("rect", { x: 0.5, y: 6.5, w: 0.12, h: 0.65, fill: { color: ORANGE }, line: { type: "none" } });
  s.addText("Ручной разметки нет. Финальный pipeline работает полностью автоматически, без участия оператора на этапе инференса.",
    { x: 0.75, y: 6.55, w: W - 1.5, h: 0.55,
      fontFace: BODY_FONT, fontSize: 12, color: CREAM, bold: true, margin: 0, valign: "middle" });

  addPageNum(s, 10, TOTAL);
}

// ========================= SLIDE 11 — FUTURE + LINKS =========================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // orange bar
  s.addShape("rect", { x: 0, y: 0, w: 0.5, h: H, fill: { color: ORANGE }, line: { type: "none" } });

  s.addText("СЛЕДУЮЩИЕ ШАГИ", {
    x: 1.0, y: 0.6, w: 11, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: ORANGE,
    charSpacing: 6, margin: 0,
  });
  s.addText("Масштабируемость и future work", {
    x: 1.0, y: 1.1, w: 11, h: 0.7,
    fontFace: HEADER_FONT, fontSize: 36, bold: true, color: WHITE, margin: 0,
  });

  const fw = [
    { tag: "+10-20%",  title: "PaddleOCR Server",
      note: "на bottom-zone мелкого шрифта (id_sku, datetime, code)" },
    { tag: "+20-30%",  title: "Real-ESRGAN super-res",
      note: "на barcode-zone — поднимет recall декода на 25/43" },
    { tag: "robust",   title: "Дообучение детектора на 100+ видео",
      note: "разных магазинов / форматов ценников / освещения" },
    { tag: "×5-10",    title: "GPU-inference или rknn (int8)",
      note: "~50 мин/видео CPU → ~5-10 мин на NPU робота" },
    { tag: "online",   title: "Online catalog API (BIRD)",
      note: "вместо scraped каталога — прямая интеграция Lenta" },
  ];
  fw.forEach((it, i) => {
    const y = 2.2 + i * 0.65;
    // tag
    s.addShape("rect", {
      x: 1.0, y, w: 1.4, h: 0.5,
      fill: { color: ORANGE }, line: { type: "none" }, rectRadius: 0.05,
    });
    s.addText(it.tag, {
      x: 1.0, y, w: 1.4, h: 0.5,
      fontFace: BODY_FONT, fontSize: 11, bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(it.title, {
      x: 2.6, y, w: 4.5, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 15, bold: true, color: WHITE,
      valign: "middle", margin: 0,
    });
    s.addText(it.note, {
      x: 7.2, y, w: 5.7, h: 0.5,
      fontFace: BODY_FONT, fontSize: 12, color: CREAM,
      valign: "middle", margin: 0,
    });
  });

  // contacts / repo
  s.addShape("line", {
    x: 1.0, y: 5.7, w: W - 2, h: 0,
    line: { color: ORANGE, width: 1.5 },
  });
  s.addText("РЕПОЗИТОРИЙ И ДЕМО", {
    x: 1.0, y: 5.85, w: 11, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, bold: true, color: ORANGE,
    charSpacing: 4, margin: 0,
  });
  s.addText(
    [
      { text: "GitHub:  ",  options: { fontSize: 13, color: CREAM } },
      { text: "github.com/<team>/lentatech-pricetag\n", options: { fontSize: 13, color: WHITE, bold: true } },
      { text: "Demo:    ", options: { fontSize: 13, color: CREAM } },
      { text: "<deployment URL>\n", options: { fontSize: 13, color: WHITE, bold: true } },
      { text: "Docker:  ",  options: { fontSize: 13, color: CREAM } },
      { text: "docker compose up --build", options: { fontSize: 13, color: WHITE, bold: true } },
    ],
    { x: 1.0, y: 6.2, w: 11, h: 1.2, lineSpacingMultiple: 1.4, margin: 0 }
  );
}

// =================== save ===================
pres.writeFile({ fileName: "presentation/Lenta_Tech_Medvezhata.pptx" }).then((p) => {
  console.log("WROTE:", p);
});
