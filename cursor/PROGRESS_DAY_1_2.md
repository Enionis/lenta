# Прогресс: День 1-2 ML/CV

## Выполнено

### День 1: Анализ данных и подготовка
- [x] Создано виртуальное окружение (`venv/`)
- [x] Установлены зависимости (requirements_ml.txt)
- [x] Создана структура ML модуля:
  ```
  backend/ml/
  ├── src/
  │   ├── detection/      # YOLO детекция
  │   ├── ocr/            # OCR (placeholder)
  │   ├── qr/             # QR (placeholder)
  │   ├── tracking/       # ByteTrack
  │   ├── utils/          # Утилиты
  │   └── pipeline.py     # Основной пайплайн
  ├── notebooks/          # Анализ и обучение
  ├── models/             # Модели
  ├── data/               # Данные
  └── cli.py              # CLI интерфейс
  ```
- [x] Проанализирована CSV разметка:
  - **157 ценников** в 3 CSV файлах
  - **156 уникальных товаров**
  - Поля: filename, product_name, price_default, barcode, и др.
  - Размеры bbox: ~160-380px width, ~200-540px height
  - Время: 0 - 88 секунд
  - Цвета: 156 red, 1 yellow

### День 2: Детекция
- [x] Реализован `PriceTagDetector` (YOLOv8n)
- [x] Интегрирован ByteTrack для трекинга
- [x] Реализован выбор лучших кадров (Лапласиан + площадь)
- [x] Создана интеграция с backend (`pipeline.py`)
- [x] Создан CLI (`python -m ml.cli --video ...`)

## Результаты теста

```bash
$ python test_ml.py

Testing imports...
[OK] pipeline import
[OK] detector import
[OK] video_utils import

Testing detector...
[OK] Detector loaded
[OK] Empty frame detection: 0 detections

Testing pipeline...
Found video: Данные\25_12-20\25_12-20.mp4
  0%: Инициализация ML pipeline...
  10%: Детекция ценников...
  ...
  100%: Готово!

[OK] Pipeline completed
  Detections: 4
  CSV: test_outputs\results.csv
  Preview: test_outputs\preview.jpg
```

## Структура CSV выхода (30 полей)

| Поле | Статус |
|------|--------|
| filename | ✅ |
| frame_timestamp, x_min, y_min, x_max, y_max | ✅ |
| color | ✅ (hardcoded: red) |
| product_name, price_default, barcode, ... | 🔄 OCR в дни 8-10 |
| QR поля | 🔄 День 11 |

## Следующие шаги

### День 3-5: Дообучение YOLO
- [ ] Создать датасет из разметки
- [ ] Дообучить YOLOv8n на ценниках
- [ ] Экспорт в ONNX

### День 6-7: Оптимизация трекинга
- [ ] Улучшить выбор лучших кадров
- [ ] Дедупликация ценников

### День 8-10: OCR
- [ ] Интегрировать PaddleOCR
- [ ] Парсинг полей ценника

### День 11-14: QR и цвет
- [ ] QR/Barcode распознавание
- [ ] Определение цвета (HSV)
- [ ] Special symbols (К/Ш)

## Запуск

```bash
# Активировать окружение
.\venv\Scripts\activate

# Тест ML
python test_ml.py

# Обработка видео
python -m ml.cli --video Данные\25_12-20\25_12-20.mp4 --output results.csv

# Запуск backend
python -m backend.app.main
```

## Модели

| Модель | Расположение | Размер |
|--------|--------------|--------|
| YOLOv8n | backend/ml/models/yolov8n.pt | 6.2 MB |

## Примечания

- Детекция работает на COCO-pretrained модели
- Для лучшей точности нужно дообучение (дни 3-5)
- OCR и QR пока заглушки (будут реализованы в следующие дни)
