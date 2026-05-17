# Интеграция ML pipeline с Backend

## TL;DR для backend-разработчика

```python
from ml.pipeline import PriceTagPipeline

pipe = PriceTagPipeline()  # загружает YOLO + LLM один раз (~5 сек)

# в обработчике загрузки видео:
df = pipe.process_video("/path/to/uploaded_video.mp4")
df.to_csv("result.csv", index=False, encoding="utf-8")
# готово — отдать пользователю result.csv
```

Один `PriceTagPipeline()` объект на процесс — модель грузится один раз.
`process_video()` потокобезопасен **не** на одном объекте (YOLO state), но
отдельные процессы можно запускать параллельно.

---

## Что внутри pipeline

```
видео (.mp4)
  ↓
[1] Tiled YOLO inference (2×2 patches + global, conf=0.15)
    + greedy IoU-трекер + top-K=3 best frames на трек
  ↓
[2] Hi-res crop из 4K кадра (padding 30%)
[3] Extended QR-crop (pad_y_top=120%, QR над ценником)
  ↓
[4] EasyOCR (ru+en, MPS/CUDA/CPU) на hi-res + tile-OCR на нижней зоне
  ↓
[5] Декод штрихкодов/QR — 3 прохода:
    • WeChat QR detector на QR-crop'ах,
    • WeChat на ПОЛНОМ 4K кадре,
    • aggressive: CLAHE+Otsu+upscale + pyzbar/zxing/cv2.barcode
  ↓
[6] Парсер полей (цены, скидка, sku, datetime, color, additional_info)
[7] Парсер QR-данных (11 полей)
[8] LLM (Qwen2.5-3B Q4 локально) → product_name
[9] Дедуп по bbox+ts + фильтр «должна быть цена/код/читаемый текст»
  ↓
DataFrame: 29 колонок по ТЗ
```

Время обработки одного видео: **30–60 минут** (CPU+MPS). OCR — bottleneck.

---

## API

### `PriceTagPipeline(...)` — конструктор

```python
PriceTagPipeline(
    weights="ml/output/runs/pricetag_v2/weights/best.pt",  # YOLO
    conf=0.15,            # confidence threshold (для tiling-режима)
    imgsz=960,            # YOLO inference size
    frame_step=3,         # каждый N-й кадр (3 — оптимум)
    top_k=3,              # сколько best frames на трек
    use_llm=True,         # выключить если LLM не нужен
    gguf_path=None,       # путь к Qwen GGUF (по умолчанию ml/weights/qwen2.5-3b-q4_k_m.gguf)
    workdir=None,         # папка для intermediate files (None = temp, авто-удаление)
)
```

### `pipe.process_video(video_path) -> pd.DataFrame`

Возвращает DataFrame с **29 колонками по ТЗ** в правильном порядке:

```
filename, product_name, price_default, price_card, price_discount,
barcode, discount_amount, id_sku, print_datetime, code, additional_info,
color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price,
wholesale_level_2_count, wholesale_level_2_price,
action_price_qr, action_code_qr
```

Семантика по ТЗ:
- Поле, не предусмотренное типом ценника → `"нет"`.
- Поле, которое пытались распознать, но не получилось → пусто.

---

## Требования к окружению

### Зависимости Python (уже в .venv)

```
ultralytics>=8
easyocr
opencv-contrib-python
torch
pandas, numpy, tqdm, certifi
zxing-cpp
pyzbar          # требует системный libzbar
llama-cpp-python  # для LLM
```

### Системные зависимости

**macOS:**
```bash
brew install zbar  # для pyzbar
```

**Linux:**
```bash
apt-get install libzbar0
```

**Без zbar pipeline всё равно работает** — pyzbar просто не используется,
остаются WeChat + zxing + cv2.barcode.

### Веса (артефакты репозитория)

```
ml/output/runs/pricetag_v2/weights/best.pt    # YOLO детектор (~6 MB)
ml/weights/qwen2.5-3b-q4_k_m.gguf             # LLM (~1.8 GB)
ml/weights/wechat_qr/{detect,sr}.{prototxt,caffemodel}  # WeChat QR (~1 MB)
```

**Без LLM-весов** pipeline работает с `use_llm=False`, product_name остаётся пустым.
**Без WeChat-весов** просто пропускается соответствующий проход (WeChat skipped).

---

## CLI

```bash
# базовый запуск
python3 ml/pipeline.py /path/to/video.mp4 --out result.csv

# без LLM (быстрее, product_name пустой)
python3 ml/pipeline.py video.mp4 --out result.csv --no_llm

# с явной рабочей директорией (для отладки)
python3 ml/pipeline.py video.mp4 --out result.csv --workdir /tmp/pipeline_run
```

---

## Интеграция с UI (Streamlit пример)

```python
import streamlit as st
import tempfile
from ml.pipeline import PriceTagPipeline

@st.cache_resource
def get_pipeline():
    return PriceTagPipeline()

st.title("Lenta — распознавание ценников")
file = st.file_uploader("Видео с робота", type=["mp4"])
if file:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(file.read())
        video_path = f.name
    with st.spinner("Обработка..."):
        df = get_pipeline().process_video(video_path)
    st.success(f"Распознано {len(df)} ценников")
    st.dataframe(df.head(20))
    st.download_button("Скачать CSV",
                       df.to_csv(index=False, encoding="utf-8"),
                       "result.csv", "text/csv")
```

Gradio — аналогично.

---

## Если backend на отдельной машине без GPU/MPS

Pipeline сам выбирает device (CUDA → MPS → CPU). На CPU скорость падает
~3-5×. Веса под int8 (RKNN) уже планируются — день 16+.

Если LLM слишком медленный — отключите его (`use_llm=False`):
- с LLM: ~50 мин на 60-сек видео,
- без LLM: ~35 мин,
- product_name остаётся пустым.

---

## Бэкап моделей (для совместного развития)

Перед переобучением `train_yolo.py` **автоматически** делает бэкап
старых весов в `ml/output/runs/_backup/<name>__<timestamp>/`.

Ручной бэкап:
```bash
python3 ml/scripts/backup_weights.py backup pricetag_v2
```

Список:
```bash
python3 ml/scripts/backup_weights.py list
python3 ml/scripts/backup_weights.py list --name pricetag_v2
```

Откат:
```bash
python3 ml/scripts/backup_weights.py restore pricetag_v2__20260515_193233
# (если есть текущий v2 — он автоматом бэкапнется перед заменой)
```

---

## Текущие метрики решения

По официальной метрике организаторов (доля ценников из GT, у которых
≥80% содержательных полей распознано правильно):

```
=== ИТОГО ===  TARGET_METRIC = 1/157 = 0.006
```

(на 26_12-20: 1 ценник с score 19/23 = 82.6%)

Avg per-pair score на сматченных треках: **0.43–0.46** (близко к
порогу 0.80, но цены/штрихкоды на этих видео часто нечитаемы из-за
fisheye + motion blur).

Подробные отчёты по дням: `ml/DAY1_REPORT.md` … `ml/DAY15_REPORT.md`.

---

## Что НЕ входит в pipeline.py и почему

| Шаг | Где живёт | Почему не в pipeline.py |
|---|---|---|
| Тренировка модели | `ml/scripts/train_yolo.py` | долгая, один раз |
| Сборка датасета | `ml/scripts/build_gt_dataset*.py` | one-off утилита |
| Эвалюация vs GT | `ml/scripts/evaluate_official.py` | offline-только |
| Бэкап моделей | `ml/scripts/backup_weights.py` | dev-утилита |

---

## Контакт

ML-роль: вопросы по pipeline / переобучению модели / парсерам полей.
Backend-роль: UI, Docker, деплой, очередь задач.
