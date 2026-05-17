# Day 13 — Tiling recall + WeChat QR + расширенные QR-crop'ы

## Главный инсайт дня

Глядя на `pseudo_vis/`, увидел: **bbox от детектора ловит только КРАСНУЮ часть ценника (где цена), а QR-код висит НИЖЕ на белом фоне** и обрезается. Все попытки декодировать QR на стандартных crop'ах обречены — мы декодируем пустоту.

## Что сделано

### 1. WeChat QR detector (ML-based)
- Веса каффе-моделей скачаны в `ml/weights/wechat_qr/` (~1 MB):
  - `detect.prototxt`, `detect.caffemodel` — детектор
  - `sr.prototxt`, `sr.caffemodel` — super-resolution для маленьких QR
- Загружается через `cv2.wechat_qrcode_WeChatQRCode()`.
- На crop'ах от старого pipeline ничего не нашёл (QR обрезан).

### 2. `extract_qr_crops.py` — расширенные crop'ы специально под QR
- Для каждой строки `track_summary_hires`:
  - Если ценник вертикальный (h ≥ w) → расширяем **вниз на 120%** высоты bbox, вверх на 40%.
  - Если горизонтальный → симметрично ±40-60%.
- Сохраняем в `best_frames_qr/<stem>/track_NNNN_rN.jpg`.
- На таких crop'ах QR-код **полностью попадает в кадр** (см. пример в `best_frames_qr/26_12-20/`).

### 3. `decode_qr_crops.py` — декод с WeChat + pyzbar + zxing на расширенных crop'ах
- WeChat (без поворотов и upscale — он сам справляется).
- Pyzbar/zxing на CLAHE+Otsu вариантах с ×2/×4 upscale и 4 ориентациями.

### 4. `wechat_wholeframe.py` — WeChat на полном 4K кадре
- Для каждого уникального `frame_idx` открываем оригинальный 4K-кадр, прогоняем WeChat, привязываем найденные QR к bbox трека (с расширением вниз на 1.5h для учёта QR-ниже-цены).
- На 26_12-20: 5/325 кадров имеют успешный декод → 4 новых QR.

### 5. `detect_and_track_tiled.py` — Tiling 2×2 + global inference
- Делим 4K-кадр на 4 patch'а с overlap 20%, прогоняем v2-детектор на каждом + ещё раз на downscaled global.
- Объединяем bbox через global NMS (IoU=0.45).
- Свой простой greedy tracker по IoU между соседними обрабатываемыми кадрами (step=3 для скорости).
- **На 25_12-20: 16 → 270 raw tracks** (после filter «должна быть цена» — отдельно). 

## Декод QR/штрихкодов: суммарно за все 4 прохода

| Видео | barcode_raw | qr_raw | OCR-rows |
|---|---|---|---|
| 25_12-20 | 2 | 1 | ~470 |
| 26_12-20 | 13 | 10 | ~480 |
| 43_15 | 2 | 0 | ~290 |

Это всё ещё **~3% строк**. Объективное ограничение качества видео — fisheye + motion blur. WeChat QR (deep-learning детектор) не помог так как ML-модели тоже не справляются с искажениями.

## Метрики до/после tiling

| Видео | Pred (12→13) | Matched (12→13) | Max per-pair |
|---|---|---|---|
| 25_12-20 | 16 → **161** | 13 → **49** | 13/23 |
| 26_12-20 | 40 → **170** | 26 → **57** | **16/23 = 0.696** |
| 43_15 | 27 → **102** | 14 → **27** | 12/23 |

**Recall детектора утроился** благодаря tiling. Знаменатель TARGET_METRIC сильно не меняется (=|GT|), но **числитель** теперь имеет шанс расти.

**На 26_12-20 best pair = 16/23 (0.70)** — близко к 80% порогу. Не хватает 3 полей.

## TARGET_METRIC

```
=== 25_12-20 ===  avg=0.431  ≥80%: 0/49  TARGET=0/57 = 0.000
=== 26_12-20 ===  avg=0.442  ≥80%: 0/57  TARGET=0/71 = 0.000
=== 43_15    ===  avg=0.461  ≥80%: 0/27  TARGET=0/29 = 0.000
=== ИТОГО    ===  TARGET = 0/157 = 0.000
```

Avg per-pair немного просел (0.46 → 0.44) — это потому что среди 49 pred-pairs (вместо 13) много с плохим OCR. Но **топ-пары стали лучше**, и до прохождения порога остаётся 2-3 поля.

## Что закрыть в дне 14, чтобы пробить 80%

Анализ топ-пары (16/23 на 26_12-20):
- **price_default, price_card**: всё ещё ловится в 5-30% — нужно дополнительный OCR на верхнюю часть crop'а.
- **id_sku, print_datetime**: 0% — мелкий текст внизу, нужен PaddleOCR с upscale ×4.
- **discount_amount**: 19-21% — regex уже расширили, но всё ещё теряем.
- **product_name**: 10-15% — LLM работает, но OCR не даёт качественных текстов.

Закрытие хотя бы 3 из этих 5 полей даст несколько пар ≥0.80 → **target_metric впервые станет > 0**.

## Стоимость прогона

- Tiling: ~10-15 мин на видео (5 inference per frame × 700-1000 кадров).
- WeChat: ~10 мин на видео.
- Aggressive decode на crop'ах: 5-8 мин на видео.
- OCR: 10-15 мин на видео (большее число crop'ов).
- LLM (product_name): 1-2 мин на видео.

**Полный pipeline на 1 видео: 30-50 минут на CPU+MPS.** Адекватно для офлайн-обработки.

## Артефакты

- `ml/scripts/extract_qr_crops.py` — расширенные QR-crop'ы
- `ml/scripts/decode_qr_crops.py` — WeChat+pyzbar+zxing на QR-crop'ах
- `ml/scripts/wechat_wholeframe.py` — WeChat на 4K-кадре
- `ml/scripts/detect_and_track_tiled.py` — tiling 2×2 + global + simple greedy tracker
- `ml/weights/wechat_qr/` — каффе-веса WeChat (~1 MB суммарно)

## Команда воспроизведения

```bash
source .venv/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/lib
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")

# 1) tiled detection
python3 ml/scripts/detect_and_track_tiled.py \
  --videos Данные/<v>/<v>.mp4 --conf 0.15 --imgsz 960 --frame_step 3

# 2) hires + qr crops
python3 ml/scripts/extract_hires_crops.py --pad_pct 0.30
python3 ml/scripts/extract_qr_crops.py

# 3) OCR
python3 ml/scripts/run_ocr_qr.py

# 4) QR decodes (4 прохода!)
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/decode_qr_crops.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/wechat_wholeframe.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/aggressive_decode.py

# 5) build + LLM + eval
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/llm_product_name.py \
    --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
done
python3 ml/scripts/evaluate_official.py
```
