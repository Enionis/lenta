# Статус дообучения OCR

## 🚀 Подготовка завершена

Создан полный pipeline для дообучения OCR.

---

## 📁 Созданные файлы

### 1. `full_ocr_training.py`
**Полный скрипт подготовки датасета**

Выполняет:
- ✅ Подготовка датасета из всех видео
- ✅ Конвертация в формат PaddleOCR
- ✅ Скачивание предобученной модели
- ✅ Создание конфигурации обучения
- ✅ Создание словаря символов

### 2. Структура данных
```
backend/ml/data/
├── ocr_dataset/              # Подготовленный датасет
│   ├── images/               # ROI ценников (~50-100 изображений)
│   └── labels/               # JSON разметка
├── paddleocr_format/         # Формат для обучения
│   ├── train/
│   │   ├── images/
│   │   └── labels.txt
│   └── val/
│       ├── images/
│       └── labels.txt
└── dataset_info.json         # Информация о датасете
```

### 3. Конфигурация обучения
```
backend/ml/
├── configs/
│   └── ocr_train.yml        # Конфигурация обучения
├── dict/
│   └── lenta_dict.txt       # Словарь символов
└── models/pretrained/       # Предобученная модель
    └── ch_PP-OCRv4_rec_train/
```

---

## 🎯 Команды для запуска обучения

### Шаг 1: Подготовка (уже выполнено)
```bash
python full_ocr_training.py
```

### Шаг 2: Установка инструментов обучения
```bash
.\venv\Scripts\activate
pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddleocr[train]
```

### Шаг 3: Запуск обучения
```bash
cd backend/ml
python -m paddleocr.tools.train -c configs/ocr_train.yml
```

**Время обучения:**
- GPU (RTX 2060): **2-4 часа**
- CPU: 10-20 часов

### Шаг 4: Экспорт модели
```bash
python -m paddleocr.tools.export_model \
    -c configs/ocr_train.yml \
    -o Global.checkpoints=output/ocr_lenta/best_accuracy \
       Global.save_inference_dir=output/ocr_lenta_inference
```

### Шаг 5: Интеграция в pipeline
Обновить `backend/ml/src/ocr/ocr_processor.py`:
```python
self.ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ru',
    rec_model_dir='backend/ml/output/ocr_lenta_inference',
    det_model_dir='backend/ml/output/ocr_lenta_inference'
)
```

---

## 📊 Ожидаемые результаты

### Текущая точность
- Детекция: **77%** (44/57 ценников)
- OCR: **0-10%** (требует дообучения)
- Цвет: **100%**

### После дообучения OCR
- OCR: **80-90%**
- Итоговая метрика: **77% × 85% = 65%** (нужно больше)

### Для достижения 80%+
Требуется:
1. Дообучить OCR (80-90% точность)
2. Улучшить детекцию до **90%+**
3. Использовать QR коды (если доступны)

---

## 🔧 Альтернатива: Улучшение без обучения

Если обучение PaddleOCR не дает результатов:

### 1. EasyOCR (попробовать)
```bash
pip install easyocr
```

```python
import easyocr
reader = easyocr.Reader(['ru', 'en'], gpu=True)
result = reader.readtext(image)
```

### 2. TrOCR (от Microsoft)
```bash
pip install transformers torch
```

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
```

### 3. Фокус на QR кодах
Если QR содержат всю информацию - не нужен OCR для основных полей.

---

## ⚠️ Важно

### Требования к системе для обучения
- **GPU**: NVIDIA с CUDA 11.8+ (RTX 2060 подойдет)
- **RAM**: 16GB+
- **Место**: 10GB свободного места
- **Время**: 2-4 часа на GPU

### Если нет GPU
Обучение на CPU займет **10-20 часов**.
Рекомендуется:
- Использовать Google Colab (бесплатный GPU)
- Или Kaggle Notebooks

---

## 📖 Инструкция для Google Colab

1. Загрузить датасет на Google Drive
2. Открыть Colab Notebook
3. Установить PaddleOCR
4. Запустить обучение
5. Скачать обученную модель

**Ссылка на Colab:** (создать notebook)
```python
# Установка
!pip install paddlepaddle-gpu
!pip install paddleocr

# Загрузка датасета
from google.colab import drive
drive.mount('/content/drive')

# Обучение
!python -m paddleocr.tools.train -c /content/drive/MyDrive/ocr_train.yml
```

---

## ✅ Что сделано

- ✅ Скрипт подготовки датасета
- ✅ Конвертация в формат PaddleOCR
- ✅ Конфигурация обучения
- ✅ Словарь символов
- ✅ Инструкции по обучению

---

## 🎯 Следующий шаг

**Запустить обучение:**
```bash
cd backend/ml
python -m paddleocr.tools.train -c configs/ocr_train.yml
```

Или использовать **Google Colab** для более быстрого обучения на GPU.

---

**Статус**: Готово к обучению ✅
**Время обучения**: 2-4 часа (GPU)
**Ожидаемая точность**: 80-90%
