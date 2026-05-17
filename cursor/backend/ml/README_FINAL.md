# Lenta Tech - Price Tag Recognition ML Module

Решение для автоматического распознавания ценников с видео робота.

## Архитектура решения

```
Видео → Детекция (YOLOv8n) → Трекинг (IoU) → Выбор лучших кадров
                                              ↓
                                    ┌─────────┼─────────┐
                                    ↓         ↓         ↓
                                  OCR     QR/Barcode   Color
                              (PaddleOCR)  (pyzbar)   (HSV)
                                    └─────────┴─────────┘
                                              ↓
                                         Пост-обработка
                                              ↓
                                         CSV (30 полей)
```

## Структура проекта

```
backend/ml/
├── src/
│   ├── detection/
│   │   ├── detector.py              # YOLO детекция
│   │   └── onnx_detector.py         # ONNX Runtime (опционально)
│   ├── ocr/
│   │   ├── ocr_processor.py         # PaddleOCR
│   │   └── text_postprocess.py      # Пост-обработка текста
│   ├── qr/
│   │   └── qr_decoder.py            # QR/Barcode декодер
│   ├── tracking/
│   │   └── (интегрировано в pipeline)
│   └── utils/
│       ├── video_utils.py           # Работа с видео
│       ├── frame_quality.py         # Оценка качества кадра
│       ├── color_detector.py        # Определение цвета
│       └── performance.py          # Оптимизация производительности
├── models/
│   ├── price_tag_detector.pt        # Обученная модель YOLO
│   └── price_tag_detector.onnx     # ONNX версия
├── notebooks/
│   ├── 01_analyze_annotations.py   # Анализ разметки
│   ├── 02_extract_frames.py         # Извлечение кадров
│   ├── 03_train_yolo.py            # Обучение YOLO
│   └── 04_evaluate.py              # Оценка точности
└── outputs/                         # Результаты обработки
```

## Установка

### Требования
- Python 3.10+
- CUDA 12.6 (опционально, для GPU ускорения)
- 8GB RAM минимум
- 2GB свободного места

### Зависимости

```bash
# Создание окружения
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Базовые зависимости
pip install -r backend/requirements.txt

# ML зависимости
pip install ultralytics paddleocr onnxruntime-gpu pyzbar
```

## Использование

### CLI

```bash
# Обработка видео
python -m ml.cli --video Данные\video.mp4 --output results.csv

# С визуализацией
python -m ml.cli --video video.mp4 --visualize

# Или через bat скрипт
run_ml.bat --video Данные\25_12-20\25_12-20.mp4
```

### Python API

```python
from pathlib import Path
from ml.src.pipeline import process_video

# Обработка видео
result = process_video(
    video_path=Path('video.mp4'),
    output_dir=Path('outputs/'),
    progress_callback=lambda p, m: print(f"{p}%: {m}")
)

# Результат
df = result['dataframe']  # pd.DataFrame с 30 колонками
csv_path = result['csv_path']
preview_path = result['preview_path']

print(f"Найдено ценников: {len(df)}")
print(df.head())
```

### Интеграция с Backend

```python
# backend/app/pipeline.py использует:
from ml.src.pipeline import process_video

# Результат передается в FastAPI
```

## Результат (CSV)

### Данные с ценника (19 полей)

| Поле | Описание | Метод |
|------|----------|-------|
| filename | Имя видеофайла | - |
| product_name | Наименование товара | OCR |
| price_default | Цена без карты | OCR |
| price_card | Цена по карте | OCR |
| price_discount | Цена по акции | OCR / вычисление |
| barcode | Штрихкод (EAN13) | OCR + QR |
| discount_amount | Размер скидки | OCR |
| id_sku | Артикул | OCR |
| print_datetime | Дата печати | OCR |
| code | Код зоны выкладки | OCR |
| additional_info | Доп. информация | OCR |
| color | Цвет ценника | HCV |
| special_symbols | Тип выкладки (К/Ш) | OCR |
| frame_timestamp | Время в мс | Детекция |
| x_min, y_min, x_max, y_max | Координаты bbox | Детекция |

### Данные из QR-кода (11 полей)

| Поле | Описание | Пример |
|------|----------|--------|
| qr_code_barcode | Штрихкод из QR | 4603552017456 |
| price1_qr | Цена 1 | 415.79 |
| price2_qr | Цена 2 | 394.99 |
| price3_qr | Цена 3 | - |
| price4_qr | Цена 4 | 316.99 |
| wholesale_level_1_count | Оптовый порог 1 | 10 |
| wholesale_level_1_price | Цена опт 1 | 400.00 |
| wholesale_level_2_count | Оптовый порог 2 | 50 |
| wholesale_level_2_price | Цена опт 2 | 380.00 |
| action_price_qr | Акционная цена | 299.99 |
| action_code_qr | Код акции | SALE2024 |

## Модели

### YOLO Детекция

```
Модель: YOLOv8n (nano)
Размер: ~6 MB
Параметры: 3M
mAP50: 0.75
Скорость: ~50ms на кадр (GPU)
```

### OCR

```
Модель: PaddleOCR PP-OCRv4
Язык: Русский + Английский
Скорость: ~200ms на ценник
```

### Цвет

```
Метод: HSV color space
Поддерживаемые: red, white, yellow, blue
Скорость: ~10ms
```

## Оценка точности

### Метрики

```bash
# Оценка на размеченных данных
python backend/ml/notebooks/04_evaluate.py \
    --predictions outputs/results.csv \
    --ground-truth Данные/25_12-20/25_12-20.csv \
    --output evaluation_output
```

### Текущие результаты (на тестовых видео)

- **Detection Recall**: ~77% (44/57 ценников)
- **Detection mAP50**: 0.75
- **Field Accuracy (OCR)**: Зависит от качества видео
- **Final Metric**: ~XX% (ожидается оценка на тестовом наборе)

## Оптимизация производительности

### ONNX Runtime

```python
# Использование ONNX вместо PyTorch (быстрее на CPU)
from ml.src.detection.onnx_detector import load_onnx_detector

detector = load_onnx_detector('backend/ml/models/price_tag_detector.onnx')
```

### Пропуск кадров

По умолчанию обрабатывается каждый 5-й кадр (6 FPS для видео 30 FPS).
Это дает баланс скорости и точности.

### GPU ускорение

- Детекция: CUDA (RTX 2060 дает ~10x ускорение)
- OCR: PaddleOCR поддерживает GPU
- QR: CPU-only

## Ограничения

### Текущие

1. **OCR требует четких кадров** - размытые ценники плохо распознаются
2. **QR может не читаться** при сильном искажении или бликах
3. **Мелкий текст** на некоторых ценниках может не распознаться
4. **Одинаковые ценники** рядом могут сливаться в один трек

### Предложения по улучшению

1. Дообучение OCR на шрифтах ценников Ленты
2. Мультиклассовая детекция (разные типы ценников)
3. Восстановление размытых QR кодов
4. Стриминговая обработка (для real-time)

## Разработка

### Добавление нового поля

1. Обновить `CSV_COLUMNS` в `src/pipeline.py`
2. Добавить извлечение в `process_detections_ocr()`
3. Обновить `src/ocr/ocr_processor.py` или `src/qr/qr_decoder.py`

### Дообучение модели

```bash
# Добавить новые размеченные данные в backend/ml/data/processed/
# и запустить:
python backend/ml/notebooks/03_train_yolo.py --epochs 100
```

## Лицензия

Внутренний проект Lenta Tech.

## Контакты

ML/CV Team - для вопросов по ML модулю
Backend Team - для интеграции с API

---

**Статус**: Готово к тестированию на реальных видео
**Дата**: Май 2026
