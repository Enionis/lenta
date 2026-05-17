# Day 2 — ML/CV: дообучение детектора + трекинг + best-frame

## Что сделано

### 1. Фикс псевдо-датасета
- Коллизия имён между `Данные/25_12-20/` и `Данные/Unlabeled/25_12-20.mp4` устранена: `stem = parent.name + '__' + stem`.
- Схлопнул `price_tag_red` + `price_tag_yellow` → один класс `price_tag` (стабильнее на маленьком наборе, в GT почти весь корпус красный).
- Yellow HSV-пороги ужесточил (S≥150, V≥160, H 22..32) — сократил шум с 3447 → 1438 жёлтых.
- Финальный датасет: **823 кадра** (714 train / 109 val), ~12.5k псевдо-bbox.

### 2. Fine-tune YOLOv8n
- `scripts/train_yolo.py`, MPS (Apple Silicon), imgsz=640, batch=16, 25 эпох, ~16 мин.
- Аугментации под наши условия: HSV-сдвиги, поворот ±10°, scale 0.4, hflip, mosaic 0.5.
- **Финальные метрики на val (псевдо-GT):** P=0.900, R=0.892, mAP@0.5=**0.956**, mAP@0.5:0.95=0.763.
- Веса: [ml/output/runs/pricetag_v1/weights/best.pt](ml/output/runs/pricetag_v1/weights/best.pt) (5.9 MB).

### 3. ByteTrack по видео + выбор лучшего кадра
- `scripts/detect_and_track.py` — `model.track(..., tracker='bytetrack.yaml', persist=True)`.
- Для каждого `track_id` копим bbox по всем кадрам, выбираем кадр с max `area × sharpness(crop)`.
- Дамп: crop'ы лучшего кадра в `output/tracks/best_frames/<video>/<track>.jpg` + summary CSV.
- На 3 размеченных видео получили 292 / 554 / 54 треков (GT: 57 / 71 / 29) — много мелких/дробных треков, фикс — повышение minimum-track-len в дне 3.

### 4. Проверка против настоящего GT
- `scripts/sanity_detector.py` — детектор без трекера, прыгаем к кадру `frame_timestamp` из GT, IoU≥0.3.
- **Recall детектора:** 25_12-20=39%, 26_12-20=55%, 43_15=52% (imgsz=960, conf=0.25).
- При imgsz=1920 conf=0.15: 38% / 55% / **72%** (43_15 заметно лучше; на остальных GT-bbox часто лежит в зоне без чёткого ценника в момент `frame_timestamp`).

### 5. Экспорт ONNX
- `best.onnx` (imgsz=960, opset=12, simplified) — **12 MB**. Готово к подключению из backend через `onnxruntime` без torch-зависимости. Путь к RKNN int8 потом через готовый ONNX → quantization → rknn-toolkit2.

## Что улучшать в дне 3

1. **Tiling**: на 4K-кадре резать на 2×2 patches с overlap, гнать YOLO на каждом → merge. Поднимет recall на мелких ценниках в 1.5–2 раза.
2. **Track filtering**: фильтр `len(track) >= 3` и NMS между треками по IoU finalbox — уберёт дубли (292→~60–80 на видео).
3. **Conf calibration**: подбор conf по результату sanity-check.
4. **Fallback на color heuristic** для кадров, где YOLO пусто (страховка recall).

## Артефакты для backend-партнёра

- Модель: `ml/output/runs/pricetag_v1/weights/best.pt` или `best.onnx`.
- Скрипт инференса: `ml/scripts/detect_and_track.py`. Возвращает CSV с полями: `track_id, frame_idx, frame_ts_ms, x_min..y_max_orig, crop_path`.
- Crop лучшего кадра уже сохраняется на диск — день 3 (OCR/QR/parser) работает прямо по crop'ам.

## Запуск

```bash
# fine-tune (если нужно переобучить)
python3 ml/scripts/train_yolo.py --epochs 25 --imgsz 640 --batch 16

# инференс по видео → треки + crop'ы + CSV
python3 ml/scripts/detect_and_track.py --videos путь/к/video.mp4 --conf 0.25

# sanity-check (требует GT csv)
python3 ml/scripts/sanity_detector.py

# экспорт в ONNX
python3 -c "from ultralytics import YOLO; YOLO('ml/output/runs/pricetag_v1/weights/best.pt').export(format='onnx', imgsz=960, simplify=True)"
```

## Интерфейс для дня 3

```python
# вход: ml/output/tracks/track_summary_<video>.csv
# для каждой строки уже есть crop_path с лучшим кадром ценника
# день 3 пишет: ocr_text(crop), qr_decode(crop), parse_fields(ocr_text) → итоговый df с колонками ТЗ
```
