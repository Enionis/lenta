# Day 1 — ML/CV: разведка + авто-разметка ценников

## Что сделано

1. **Анализ данных** (`scripts/analyze_videos.py`):
   - 6 видео по 15–89 с, все 4K (3840×2160), ~20 FPS.
   - GT-разметка: 157 ценников всего (57 + 71 + 29).
   - Цвет ценников: 99% red, 1 yellow.
   - Bbox в исходных 4K: 60–400 px по стороне (~30–220 после ресайза до 1920).
   - Поля QR заполнены почти всегда; `price_discount` / wholesale_* — почти всегда «нет».
   - Видео снято fisheye-объективом (заметные искажения по краям).
   - Робот снимает кадр повёрнутым на ~90° (стеллажи вертикально).

2. **Извлечение кадров** (`scripts/extract_frames.py`):
   - Сэмплинг каждые 0.25 с, фильтр по резкости (Лапласиан > 40) и diff с предыдущим (>3).
   - Ресайз до длинной стороны 1920 px для ускорения downstream.
   - Итог: **823 кандидатных кадра** в `output/frames/`, индекс в `output/frames_index.csv`.

3. **Авто-разметка ценников** (`scripts/pseudo_label_color.py`):
   - Сначала пробовал YOLO-World s — recall ~5% на наших кадрах. Отказался.
   - HSV-маска красного (2 полосы) + жёлтого → морфология → connectedComponents → bbox.
   - Фильтры: размер 25..400 px, аспект 0.4..3.2, заполненность маской ≥0.25, NMS 0.3.
   - Итог: **~14 600 псевдо-bbox** на 655 уникальных кадрах (collision на 168 — фикс в день 2).
   - YOLO-формат в `output/yolo_dataset/`, `data.yaml` готов для `ultralytics train`.

## Известные проблемы (для дня 2)

- **Коллизия имён** между `Данные/25_12-20/` и `Данные/Unlabeled/25_12-20.mp4`: добавить родительскую папку в stem.
- **Шум в маске**: брендинг с красным (Santo Stefano, Tomo) даёт false positives. Полагаемся на consistency между кадрами при дообучении YOLO — он отбросит редкие/случайные срабатывания.
- **Yellow false positives**: ~3500 жёлтых bbox при 1 в GT → класс yellow в датасете лучше схлопнуть с red в единый `price_tag` или поднять пороги жёлтого.

## План на день 2

1. Фикс коллизии в `pseudo_label_color.py` (stem = parent/stem).
2. Схлопнуть классы в один `price_tag` (один класс — стабильнее на маленьком наборе).
3. Fine-tune **YOLOv8n** на собранном датасете: `imgsz=960`, 60 эпох, аугментации (HSV, blur, rotate ±10).
4. ByteTrack из ultralytics → один track_id на ценник → выбор лучшего кадра (площадь × резкость).
5. Экспорт в ONNX, передача backend-партнёру.

## Интерфейс для backend (фиксируется в день 3)

```python
from ml.pipeline import PriceTagPipeline
df = PriceTagPipeline().process_video("path/to/video.mp4")
# df: pd.DataFrame с колонками из ТЗ (filename, product_name, ..., x_max, y_max,
#     qr_code_barcode, price1_qr, ..., action_code_qr)
```

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python-headless numpy pandas tqdm ultralytics certifi

python3 ml/scripts/analyze_videos.py
python3 ml/scripts/extract_frames.py
python3 ml/scripts/pseudo_label_color.py --vis 40
```
