# Day 3 — ML/CV: OCR + QR + парсер + сборка финального CSV

## Что сделано

1. **Tracker hardening** (`scripts/detect_and_track.py`):
   - фильтр коротких треков `len(track) >= 3`,
   - cross-track NMS с IoU≥0.4 (один реальный ценник = один track).
   - Результат на 26_12-20: 608 raw → 340 после фильтров (~5× от GT — есть запас).

2. **Hi-res crops из 4K** (`scripts/extract_hires_crops.py`):
   - детектор работал на ресайзе 1920, OCR на crop'ах с 1920 был пустым (ценники ~30–100 px).
   - Открываем оригинальное видео, перепрыгиваем к `frame_idx`, вырезаем bbox с padding 18%.

3. **OCR** (`scripts/run_ocr_qr.py`):
   - EasyOCR (ru+en, CPU/MPS), без онлайн API,
   - 4 ориентации crop'а (0/90/180/270), выбор по сумме confidence,
   - upscale crop'ов <200 px перед OCR.
   - На 43_15: 34/53 crop'ов содержат хоть какой-то текст; цены 99 / 679 / 159 распознаются.

4. **Штрихкод / QR** (`scripts/run_ocr_qr.py` + `whole_frame_codes.py`):
   - `cv2.barcode.BarcodeDetector` + `cv2.QRCodeDetector`,
   - дополнительный проход на ВСЁМ 4K-кадре с 4 поворотами,
   - также пробовал `zxing-cpp`.
   - **Recall = 0** на всех 3 видео — серьёзное ограничение качества видео:
     - тонкие линии штрихкода размываются motion blur'ом,
     - fisheye-искажение на краях,
     - ракурс сверху делает штрихкод трапециевидным.
   - Альтернативы для дня 4/5: `pyzbar` (нужен системный `zbar` — `brew install zbar`), Tencent QBar, специализированный ML-детектор штрихкодов.

5. **Regex-парсер полей** (`scripts/build_final_csv.py`):
   - цены: top-2 числа из OCR (price_default > price_card),
   - скидка: `[\-]?(\d{1,2})\s*%`,
   - дата печати: `dd[./]mm[./]yyyy hh[:.]mm`,
   - штрихкод: 13 подряд цифр в ОДНОМ OCR-токене (склейка из разных токенов давала шум),
   - id_sku: 9–12 цифр,
   - цвет: HSV-доминанта в crop'е.
   - Парсер QR-данных по алиасам из ТЗ (`b/barcode`, `p1/price1`, `wL1C/wholesaleLevel1Count`, ...).

6. **Финальный CSV в формате ТЗ**:
   - все 29 колонок,
   - UTF-8, запятая-разделитель, точка как десятичный,
   - «нет» для полей, не предусмотренных шаблоном,
   - пусто для нераспознанных.

7. **Единая точка входа для backend** (`ml/pipeline.py`):
   ```python
   from ml.pipeline import PriceTagPipeline
   df = PriceTagPipeline().process_video("/path/to/video.mp4")
   df.to_csv("out.csv", index=False, encoding="utf-8")
   ```

## Текущие метрики (на 3 размеченных видео)

| Видео     | Pred rows | price_default | price_card | discount | color | barcode |
|-----------|-----------|---------------|------------|----------|-------|---------|
| 25_12-20  | 172       | 18            | 3          | 0        | 27    | 0       |
| 26_12-20  | 340       | 44            | 15         | 8        | 278   | 0       |
| 43_15     | 53        | 16            | 8          | 6        | 46    | 0       |

**Barcode-match precision/recall = 0/0** — primary key не работает; матчинг ляжет на пространственно-временной ключ (`frame_timestamp` + bbox).

## Где основные потери

1. **Barcode не декодируется** — самое больное. План на день 4:
   - `pip install pyzbar` + `brew install zbar` (или статический wheel),
   - super-resolution на crop'ах штрихкода (`Real-ESRGAN`/`opencv dnn_superres`),
   - fallback: ML-детектор штрихкодов из `darknet`/`yolov5` weights.

2. **Overcount треков (5×)** — много false positives на бутылках/упаковках:
   - повысить conf до 0.4,
   - дополнительный фильтр после OCR: «если в crop'е не нашлось ни одного числа в диапазоне 1..9999, считаем не-ценником».

3. **id_sku / print_datetime = 0** — мелкий текст внизу ценника не доходит до OCR.
   - Целевой OCR-проход на нижнюю треть crop'а с upscale ×4 — будет в дне 4.

4. **product_name = пусто** — пока не парсим, нужен LLM или whitelist каталога Lenta.

## План на день 4 (для меня)

1. `pyzbar` + super-resolution → barcode recall.
2. Tile OCR (отдельные проходы на верх/середину/низ ценника).
3. Heuristic: «если pred-cluster по frame_ts_ms ± 500ms имеет ≥2 одинаковых barcode — мерджим в одну строку».
4. product_name: подключить локальный LLM (Qwen 2.5 7B Q4) для extraction по чанку OCR-текста — раз без интернета, локально.
5. Финальная подгонка conf/min_track_len под максимум F1 по полям.

## Артефакты, готовые для backend (день 4 у партнёра)

- `ml/pipeline.py` — `PriceTagPipeline().process_video(path) -> pd.DataFrame`.
- Веса: `ml/output/runs/pricetag_v1/weights/best.pt` (5.9 MB) или `best.onnx` (12 MB).
- Зависимости: `ultralytics`, `easyocr`, `opencv-contrib-python`, `pandas`, `numpy`, `tqdm`, `certifi`, `zxing-cpp`.

## Запуск end-to-end

```bash
source .venv/bin/activate
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())") \
    python3 ml/pipeline.py Данные/43_15/43_15.mp4 --out result.csv
```
