# ML Module: Price Tag Recognition

Модуль компьютерного зрения для распознавания ценников с видео робота.

## Структура

```
ml/
├── src/
│   ├── detection/          # Детекция ценников (YOLO)
│   ├── ocr/                # OCR (PaddleOCR)
│   ├── qr/                 # QR/Barcode
│   ├── tracking/           # Трекинг (ByteTrack)
│   ├── utils/              # Утилиты
│   └── pipeline.py         # Основной пайплайн
├── notebooks/              # Исследования
│   ├── 01_analyze_annotations.py
│   ├── 02_extract_frames.py
│   └── 03_train_yolo.py
├── models/                 # Модели
├── data/                   # Данные
│   ├── raw/                # Видео
│   └── processed/          # Обработанные кадры
├── outputs/                # Результаты
├── cli.py                  # CLI интерфейс
└── README.md               # Этот файл
```

## Установка

```bash
# Виртуальное окружение
python -m venv venv
.\venv\Scripts\activate  # Windows
# или
source venv/bin/activate  # Linux/Mac

# Зависимости
pip install -r requirements_ml.txt
```

## Использование

### CLI

```bash
# Обработка видео
python -m ml.cli --video path/to/video.mp4 --output results.csv

# С визуализацией прогресса
python -m ml.cli --video video.mp4 --visualize
```

### Как библиотека

```python
from ml import process_video
from pathlib import Path

result = process_video(
    video_path=Path('video.mp4'),
    output_dir=Path('outputs/'),
    progress_callback=lambda p, m: print(f"{p}%: {m}")
)

df = result['dataframe']  # pd.DataFrame с результатами
csv_path = result['csv_path']
preview_path = result['preview_path']
```

## Результат (CSV)

30 полей согласно заданию:

**Данные с ценника:**
- `filename` - имя видео
- `product_name` - наименование товара
- `price_default` - цена без карты
- `price_card` - цена по карте
- `price_discount` - цена по акции
- `barcode` - штрихкод
- `discount_amount` - размер скидки
- `id_sku` - артикул
- `print_datetime` - дата печати
- `code` - код зоны выкладки
- `additional_info` - доп. информация
- `color` - цвет ценника (red/white/yellow/blue)
- `special_symbols` - тип выкладки (К/Ш)
- `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max` - координаты

**Данные из QR:**
- `qr_code_barcode`, `price1_qr`, `price2_qr`, `price3_qr`, `price4_qr`
- `wholesale_level_1_count`, `wholesale_level_1_price`
- `wholesale_level_2_count`, `wholesale_level_2_price`
- `action_price_qr`, `action_code_qr`

## Пайплайн

```
Видео → Детекция (YOLO) + Трекинг (ByteTrack)
      ↓
Выбор лучших кадров (Лапласиан)
      ↓
OCR (PaddleOCR) + QR (pyzbar) + Color Detection
      ↓
Парсинг полей → CSV
```

## Обучение модели

```bash
# Анализ разметки
python backend/ml/notebooks/01_analyze_annotations.py

# Извлечение кадров (когда есть видео)
python backend/ml/notebooks/02_extract_frames.py

# Обучение YOLO
python backend/ml/notebooks/03_train_yolo.py --epochs 100 --model n

# Быстрый вариант (COCO pretrained)
python backend/ml/notebooks/03_train_yolo.py --quick
```

## Архитектура моделей

| Компонент | Модель | Размер |
|-----------|--------|--------|
| Детекция | YOLOv8n | ~6 MB |
| OCR | PaddleOCR mobile | ~10 MB |
| QR/Barcode | pyzbar | ~0 MB |

## Интеграция с Backend

Функция `process_video()` из `ml/src/pipeline.py` импортируется в `backend/app/pipeline.py`.

Возвращает:
```python
{
    'dataframe': pd.DataFrame,  # Результаты
    'csv_path': Path,          # Путь к CSV
    'preview_path': Path,      # Путь к превью
    'stats': {
        'processing_time': float,
        'detections_found': int,
        'results_count': int
    }
}
```

## Требования

- Python 3.10+
- RAM: 4GB минимум
- CPU: поддерживается
- GPU: опционально (ускоряет OCR)

## Ограничения

- Требуется дообучение YOLO на ценниках для лучшей точности
- OCR требует четких кадров (размытые = плохое распознавание)
- QR может не читаться при сильном искажении

## Roadmap

- [x] День 1: Анализ данных, структура
- [x] День 2: Детекция (YOLO), трекинг
- [ ] День 3-5: Дообучение YOLO
- [ ] День 6-7: Выбор лучших кадров
- [ ] День 8-10: OCR (PaddleOCR)
- [ ] День 11: QR/Barcode
- [ ] День 12-14: Color detection, парсинг
- [ ] День 15-20: Оптимизация, тестирование

## Лицензия

Внутренний проект Lenta Tech.
