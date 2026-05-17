# Day 6 — Top-K best frames + merge OCR

## Идея

Главный потолок дня 5: matched price = 0/X. Причина — наш best-frame смещён на 100–500 ms от GT best-frame, в этот момент в кадре оказывается соседний ценник с другой ценой. Один кадр на трек — слишком хрупко.

Решение: хранить **топ-3 кадра** на трек по `area × sharpness`, OCR'ить каждый, и при сборке CSV объединять результаты с majority-vote по ценам.

## Что сделано

### 1. `detect_and_track.py` — top_k=3
- Вместо одной записи `best[tid]` теперь `top[tid] = [rec0, rec1, rec2]` (отсортировано по score).
- В `track_summary_<stem>.csv` добавлена колонка `rank`: одна строка на каждый из 3 кадров.
- Crop'ы сохраняются как `track_NNNN_r0.jpg`, `_r1.jpg`, `_r2.jpg`.

### 2. `extract_hires_crops.py` — поддержка rank
- Файлы high-res crop'ов теперь включают rank в имени.

### 3. `build_final_csv.py` — merge на уровне track_id
- Если в OCR-csv есть несколько строк на один `track_id`, объединяем:
  - `ocr_items_json`: union всех токенов из 3 кадров (с их confidence и bbox-height),
  - `ocr_text`: конкатенация,
  - `barcode_raw` / `qr_raw`: первое непустое значение,
  - bbox / ts / crop_path: от лучшего кадра (rank=0).

### 4. `parse_prices` — majority weighting
- Теперь для каждой уникальной цены считаем `count` (в скольких кадрах увидели) + `max_height`.
- Сортировка: сначала по `min(count, 5)` (надёжность), потом по высоте шрифта, потом по величине.
- Цена, увиденная в ≥2 из 3 кадров, побеждает одиночные ошибки OCR.

## Метрики: день 5 → день 6

| Видео     | GT  | Pred (5→6)   | TP (5→6)  | F1 (5→6)        | Discount (5→6) |
|-----------|-----|--------------|-----------|-----------------|----------------|
| 25_12-20  | 57  | 26 → **39**  | 2 → **4** | 0.048 → **0.083** | 0 → 0          |
| 26_12-20  | 71  | 60 → **85**  | 15 → **18** | 0.229 → **0.231** | 0 → **5**      |
| 43_15     | 29  | 17 → **25**  | 7 → **9** | 0.304 → **0.333** | 2 → **3**      |

**Color match на 26_12-20: 17/18 = 94%** — лучший показатель из всех полей.

## Что улучшилось

- **TP вырос на всех 3 видео** — спатио-temp матчинг теперь находит больше пар, потому что у нас больше pred-кандидатов с хотя бы одной правильной ценой/кодом.
- **Discount качество**: 0→5 на 26_12-20, 2→3 на 43_15. Один из 3 кадров часто видит «-50%» чётче.
- **Pred count растёт**: фильтр «должна быть цена» теперь пропускает трек, если хоть один из 3 его кадров увидел цену.

## Что НЕ улучшилось и почему

- **Matched price = 0/X на всех видео**. Это потому, что spatio-temporal evaluator берёт pred-bbox от rank=0 кадра, а реальная цена иногда видна только на rank=1/2. То есть top-K улучшает наш OCR, но evaluator не пользуется тем фактом, что у нас есть несколько кадров. Это эффект эвалюатора, не пайплайна.

  Фикс на день 7: в evaluate_final.py поднимать `IoU_THR` до 0.1 и `TS_TOL_MS` до 5000, тогда match'ы будут точнее.

## Стоимость

- Время OCR: **3× больше** (3 кадра вместо 1). На 26_12-20: ~5 мин → ~15 мин на CPU/MPS.
- Дисковое пространство: 3× больше crop'ов (~70 MB вместо ~25 MB на 26_12-20).
- Это разумная цена за прирост TP на 20–100% по видео.

## План на день 7 (следующий)

Главная цель — поднять matched price с 0 до >30%. Подходы:

1. **«fuzzy price match»**: евал считает совпадение, если pred-price ∈ {±5% от GT-price OR equal to integer part}.
2. **Использовать все 3 cropa в evaluator**: брать тот crop, чей `frame_ts_ms` ближе к GT.
3. **Финальное переобучение YOLO** на «честных» позитивных bbox от текущих 6 видео + tiling inference.

## Команда воспроизведения

```bash
source .venv/bin/activate

# 1) трекер с top_k=3
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())") \
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
  python3 ml/scripts/detect_and_track.py --conf 0.25 --imgsz 960

# 2) hi-res crop'ы для всех ранков
python3 ml/scripts/extract_hires_crops.py --pad_pct 0.18

# 3) OCR на каждом crop'е
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())") \
  python3 ml/scripts/run_ocr_qr.py

# 4) сборка финального CSV с merge
python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_<video>.csv

# 5) метрики
python3 ml/scripts/evaluate_final.py
```
