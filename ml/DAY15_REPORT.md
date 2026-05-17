# Day 15 — Эксперимент с v3 (расширенный bbox) и возврат к v2

## TL;DR

- Попытка v3 (детектор обучен на bbox + QR-зоной сверху) → **регресс 1/157 → 0/157**.
- Откатились на v2 → восстановили **1/157 = 0.006** (последний рабочий рекорд).
- Дополнительный декод штрихкодов на v2: +14 barcode и +8 QR на 26_12-20 (avg per-pair 0.448 → **0.452**).

## Гипотеза v3

Из pseudo_vis было видно: bbox v2 ловит только цветную часть ценника, **QR-код висит сверху** на белом фоне и обрезается. Гипотеза: если обучить детектор на расширенных bbox'ах, чтобы он сразу выдавал bbox с QR-зоной, OCR будет работать лучше.

**Что сделано:**
1. `build_gt_dataset_extended.py`: pad_top=100% (вверх на высоту bbox), pad_bottom=30%, pad_x=30%. Получили 179 кадров × 1044 расширенных bbox.
2. Fine-tune v3 от v2 на расширенном датасете, 25 эпох. Метрики: P=0.887, R=0.868, mAP@0.5=0.941.
3. Tiled inference v3 → 203/322/90 треков.
4. Полный pipeline: hires (pad 0.30) → OCR → QR decode → build → LLM → eval.

## Результат v3: **0/157 = 0.000** — РЕГРЕСС

Причина выявлена через анализ avg per-pair: 0.448 (v2) → 0.439 (v3).

**Проблема:** v3 уже выдаёт bbox с QR-зоной (+100% по высоте). Наш `extract_hires_crops.py` добавляет ещё **pad_pct=0.30**. Итого crop = bbox v2 × 1.3 (по ширине) × 2.3 (по высоте). OCR на таком crop'е захватывает много фона, главная цена (~10-15% площади crop'а) распознаётся хуже.

**Попытка fix:** уменьшил pad_pct до 0.05 для v3 → всё равно 0/157. Видимо более широкий bbox сам по себе ухудшает OCR-фокус на цене даже без дополнительного padding.

## Возврат к v2 — финальная метрика

Полностью переснял pipeline с v2 (270/251/171 треков). Декод штрихкодов через 3 прохода:

| Видео | decode_qr_crops | wholeframe WeChat | aggressive | Σ новых |
|---|---|---|---|---|
| 25_12-20 | 0+0 | 0 QR | 1+1 | 1 barcode, 1 QR |
| 26_12-20 | 3+2 | 0 QR | 11+6 | **14 barcode, 8 QR** |
| 43_15 | 0+0 | 0 QR | 0+0 | 0 |

После всех декодов и `build_final` + LLM + `evaluate_official`:

```
=== 25_12-20 ===  avg=0.431  ≥80%: 0/49  TARGET=0/57 = 0.000
=== 26_12-20 ===  avg=0.452  ≥80%: 1/57  TARGET=1/71 = 0.014
=== 43_15    ===  avg=0.461  ≥80%: 0/27  TARGET=0/29 = 0.000
=== ИТОГО    ===  TARGET = 1/157 = 0.006
```

avg per-pair на 26_12-20: **0.448 → 0.452** (+0.4% после большего числа декодов).

## Уроки дня

1. **Расширение bbox в обучении ≠ улучшение OCR.** Детектор учится выдавать большие боксы, но OCR требует **фокусный** crop без лишнего фона. Идеальная стратегия: детектор выдаёт точный bbox, а OCR работает на цена-зоне отдельно от QR-зоны.

2. **Hybrid pipeline — правильное направление.** На день 16+: использовать v2 для основного OCR + дополнительный pass v3 (или просто `extract_qr_crops.py`) только для декода штрихкода/QR. Это уже отчасти сделано через `extract_qr_crops.py` (отдельные crop'ы для QR с pad_y_top=120%).

3. **Bottleneck — НЕ детектор, а OCR на главных полях.** На сматченных парах:
   - id_sku 0% — OCR недосчитывает 1-3 цифры
   - print_datetime 0% — нижняя зона мелкого текста не читается
   - price_default/_card 5-30% — даже на хорошем crop'е цена путается

   Закрытие этих полей требует **выделенного OCR-движка** (PaddleOCR Server) или **специализированного декодера цифр**.

## Финальная сводка за 15 дней работы

| День | Что добавлено | TARGET_METRIC |
|---|---|---|
| 1-5 | Базовый pipeline (YOLO+EasyOCR) | — |
| 6 | Top-K best frames + merge | — |
| 7 | OCR-токен cleanup, alt-bbox matching | — |
| 8 | Retrain v2 на GT bbox | — |
| 9 | Tighter padding, color center | — |
| 10 | LLM product_name (LM Studio) | — |
| 11 | **Официальная метрика реализована**, embedded llama-cpp | **0/157 = 0.000** |
| 12 | aggressive_decode QR/barcode | 0/157 |
| 13 | Tiling 2×2 + WeChat QR + extended QR crops | 0/157 |
| 14 | **Bug fixes** (ean .0, parse_qr empty) + barcode fallback + fuzzy additional_info | **1/157 = 0.006** |
| 15 | v3 эксперимент (регресс) + возврат к v2 + ещё декодов | **1/157 = 0.006** |

## Что попробовать дальше (день 16+)

1. **PaddleOCR PP-OCRv4 mobile** на нижнюю зону crop'а — может решить id_sku и print_datetime сразу. Локально (~10 MB), пройдёт ТЗ.
2. **Hybrid v2+v3:** v2 для main OCR, v3 как QR-zone proposer.
3. **Super-resolution на barcode-zone** (Real-ESRGAN local) — должно поднять recall QR-декода до 20-30%.
4. **Расширить keyword-парсеры:** code (regex `\d+_\d+`), special_symbols (LLM-based single letter extraction).
5. **product_name через RAG с каталогом Lenta** (если организаторы выложат) — exact-match лучше LLM-фантазии.

## Артефакты

- `ml/scripts/build_gt_dataset_extended.py` — генератор расширенного датасета (для возможного hybrid'а)
- `ml/output/runs/pricetag_v3/` — обученные веса v3 (на запас)
- `ml/output/final_*.csv` — финальные submission'ы (v2-based, **1 ценник прошёл порог**)
- `ml/output/final_eval_*.csv` — eval-копии с `alts_json`
- `ml/DAY11-15_REPORT.md` — отчёты по дням

## Команда воспроизведения финала

```bash
source .venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")

# tiled v2
python3 ml/scripts/detect_and_track_tiled.py \
  --weights ml/output/runs/pricetag_v2/weights/best.pt \
  --conf 0.15 --imgsz 960 --frame_step 3

python3 ml/scripts/extract_hires_crops.py --pad_pct 0.30
python3 ml/scripts/extract_qr_crops.py --pad_y_top 1.20 --pad_y_bot 0.40
python3 ml/scripts/run_ocr_qr.py
python3 ml/scripts/decode_qr_crops.py
python3 ml/scripts/wechat_wholeframe.py
python3 ml/scripts/aggressive_decode.py

for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv
  python3 ml/scripts/llm_product_name.py --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/llm_product_name.py --final ml/output/final_$v.csv --ocr ml/output/ocr_qr_$v.csv
done
python3 ml/scripts/evaluate_official.py
```
