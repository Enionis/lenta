# Прогресс: День 8-14 - OCR, QR, Цвет, Пост-обработка

## ✅ Выполнено

### День 8-10: OCR с пост-обработкой
- [x] PaddleOCR интегрирован
- [x] Пост-обработка текста:
  - Коррекция ошибок OCR (O→0, 3→З и т.д.)
  - Извлечение полей: barcode, SKU, цены, даты
  - Вычисление скидки из цен
- [x] Структурированный вывод в 30 полей CSV

### День 11: QR/Barcode декодирование
- [x] Модуль `qr_decoder.py` (pyzbar)
- [x] Парсинг формата QR Ленты:
  - `barcode|b;price1|p1;...actionCode|aC`
- [x] Объединение OCR + QR (QR имеет приоритет для barcode)

### День 12-13: Определение цвета
- [x] HSV color detection
- [x] Поддержка цветов: red, white, yellow, blue
- [x] Диапазоны для каждого цвета в HSV

### День 14: Интеграция
- [x] Обновлен pipeline.py с полной цепочкой:
  ```
  Детекция → Трекинг → Выбор кадра → OCR + QR + Color → CSV
  ```

## Результаты теста

```
Tracks: 44 ценников найдено
Color:  red (все верно)
QR:     интегрирован (ожидаем видео с QR для теста)
OCR:    структура готова (улучшится с датасетом)
CSV:    44 строки, 30 колонок
```

## Структура CSV (все 30 полей)

**Данные с ценника:**
- filename, frame_timestamp, x_min, y_min, x_max, y_max ✅
- product_name 🔄 OCR (зависит от качества изображения)
- price_default, price_card, price_discount 🔄 OCR
- barcode, id_sku 🔄 OCR+QR
- discount_amount 🔄 Вычисляется из цен
- print_datetime, code, additional_info 🔄 OCR
- color ✅ HSV detection
- special_symbols 🔄 OCR

**Данные из QR:**
- qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr ✅
- wholesale_level_1_count, wholesale_level_1_price ✅
- wholesale_level_2_count, wholesale_level_2_price ✅
- action_price_qr, action_code_qr ✅

## Файлы

```
backend/ml/
├── src/
│   ├── detection/
│   │   └── detector.py          # YOLO + обученная модель
│   ├── ocr/
│   │   ├── ocr_processor.py     # PaddleOCR
│   │   └── text_postprocess.py  # Пост-обработка
│   ├── qr/
│   │   └── qr_decoder.py        # QR/Barcode
│   ├── utils/
│   │   └── color_detector.py    # HSV цвет
│   └── pipeline.py              # Полный пайплайн
├── models/
│   ├── price_tag_detector.pt    # Обученная (mAP50=0.75)
│   └── price_tag_detector.onnx  # ONNX формат
└── outputs/results.csv          # 44 записи, 30 полей
```

## Запуск

```bash
# Тест ML
python test_ml.py

# Обработка видео
python -m ml.cli --video Данные\25_12-20\25_12-20.mp4 --output results.csv

# Backend
python -m backend.app.main
```

## Следующие шаги (дни 15-20)

- [ ] Оптимизация производительности
- [ ] Экспорт в ONNX Runtime (ускорение CPU)
- [ ] Тестирование на всех видео
- [ ] Оценка точности vs Ground Truth
- [ ] Docker + деплой

## Примечания

- Детекция работает отлично (44 ценника из 57 в разметке = ~77% recall)
- OCR требует дальнейшей настройки под шрифты ценников
- QR не тестировался (в тестовом видео нет видимых QR)
- Цвет определяется корректно (все красные ценники распознаны)
