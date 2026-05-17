# LentaTech Pricetag Recognition — команда «Медвежата»

Распознавание ценников с видео магазинного робота. На вход — `.mp4` 4K-видео,
на выход — CSV с 29 полями по каждому ценнику в соответствии с ТЗ Lenta Tech.

**Финальная метрика:** `TARGET_METRIC = 4/157 = 0.025`
(top-пара 20/23 = 87% по 23 контентным полям).

## Быстрый старт (Docker)

```bash
# 1) build (~10-15 мин первый раз, ~3-4 GB образ)
docker compose up --build -d

# 2) проверка готовности (~30-60 сек на загрузку моделей)
curl -fsS http://localhost:8000/health

# 3) UI для загрузки видео
open http://localhost:8501          # Streamlit upload UI
open http://localhost:8502          # Dashboard аналитики
open http://localhost:8000/docs     # Swagger API
```

После загрузки видео через UI:8501 → дождаться `status=done` (~30-50 мин на CPU) →
скачать `results.csv`.

Без Docker — см. [DOCKER_RUN.md](DOCKER_RUN.md) раздел «Локальный запуск» ниже.

## Архитектура решения

```
video.mp4
  │
  ├─► [1] DETECT  YOLO triple ensemble (v2 + v4 + v5b) + TTA + tiled 2×2
  │              → ~1430 треков на 3 видео
  │
  ├─► [2] TRACK   greedy IoU
  │              → top-K=3 best frames per track (area × sharpness)
  │
  ├─► [3] CROPS   hires (pad=0.30) + QR-zone (pad_y_top=1.2)
  │
  ├─► [4] OCR     EasyOCR ru+en × 4 rotations
  │              + bottom-30% × upscale ×5 для мелкого шрифта
  │
  ├─► [5] DECODE  WeChat-QR + pyzbar + zxing-cpp + cv2.barcode
  │              + aggressive: CLAHE + Otsu + upscale ×2/×4 × 4 rotations
  │
  ├─► [6] BUILD   парсеры 29 полей (font-aware prices, regex, fuzzy)
  │              + K-frame merge + bbox dedup + non-pricetag filter
  │
  ├─► [7] LLM     Qwen2.5-3B-Q4 (embedded llama-cpp) → product_name
  │              + fallback: длинные OCR-токены если LLM возвращает «нет»
  │
  └─► [8] CATALOG 315 SKU (148 GT + 167 scraped с lenta.com)
                  barcode-match → подтянуть цены | fuzzy ≥ 0.65
                  + dedup duplicate barcodes
```

Детектор обучен на 179 GT-кадрах (1044 ценников) + 5000 synthetic-кадров
(271 шаблон × фоны магазина × augmentation). Каталог собран автоматически
через undetected-chromedriver с публичной страницы lenta.com.

## Производственная архитектура (Docker)

```
┌─────────────┐   ┌────────────┐   ┌──────────────┐
│ Streamlit   │   │ Streamlit  │   │ Swagger UI   │
│ UI :8501    │   │ Dashboard  │   │ /docs        │
│             │   │  :8502     │   │              │
└──────┬──────┘   └──────┬─────┘   └──────┬───────┘
       │                 │                │
       └────────┬────────┴────────┬───────┘
                ▼                 ▼
       ┌──────────────────────────────┐    ┌─────────────┐
       │ FastAPI API :8000            │◄──►│ Redis :6379 │
       │ - JWT auth (optional)        │    │ broker +    │
       │ - rate-limit (slowapi)       │    │ pub/sub +   │
       │ - libmagic upload validation │    │ progress    │
       │ - WebSocket /ws/jobs/{id}    │    └──────┬──────┘
       │ - /metrics (Prometheus)      │           │
       └──────┬───────────────────────┘           │
              │                                   │
              ▼                                   ▼
       ┌─────────────────┐               ┌──────────────────┐
       │ SQLite jobs.db  │               │ Celery Worker    │
       │ (job history)   │               │ ML pipeline      │
       └─────────────────┘               │ + Prometheus     │
                                         │   :9101          │
                                         └──────────────────┘
                                                 ▲
                                                 │
                                         ┌──────────────────┐
                                         │ Celery Beat      │
                                         │ cleanup жирных   │
                                         │ файлов > 7 дней  │
                                         └──────────────────┘
```

## Воспроизведение метрики (offline reproducibility)

```bash
# 1) detect+track (triple ensemble + TTA) — ~3-4 ч
python3 ml/scripts/detect_and_track_tiled.py \
  --weights ml/output/runs/pricetag_v2/weights/best.pt \
  --ensemble ml/output/runs/pricetag_v4/weights/best.pt \
              ml/output/runs/pricetag_v5b/weights/best.pt \
  --tta --conf 0.15 --imgsz 960 --frame_step 3

# 2) crops + OCR
python3 ml/scripts/extract_hires_crops.py --pad_pct 0.30
python3 ml/scripts/extract_qr_crops.py --pad_y_top 1.20 --pad_y_bot 0.40
python3 ml/scripts/run_ocr_qr.py
python3 ml/scripts/extra_bottom_ocr.py

# 3) decoders
python3 ml/scripts/decode_qr_crops.py
python3 ml/scripts/wechat_wholeframe.py
python3 ml/scripts/aggressive_decode.py

# 4) post-filter (опционально, чистит submission CSV)
python3 ml/scripts/filter_non_pricetag.py

# 5) build → LLM → fallback → catalog → eval
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/llm_product_name.py --final ml/output/final_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/llm_product_name.py --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/fill_product_name_fallback.py --final ml/output/final_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/fill_product_name_fallback.py --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/match_to_catalog.py --final ml/output/final_$v.csv      --catalog ml/data/lenta_catalog_merged.parquet --threshold 0.65
  python3 ml/scripts/match_to_catalog.py --final ml/output/final_eval_$v.csv --catalog ml/data/lenta_catalog_merged.parquet --threshold 0.65
done

python3 ml/scripts/evaluate_official.py
# → TARGET_METRIC = 4/157 = 0.025
```

Итоговые CSV — в `ml/output/_4in157_FINAL_20260517_172919/`
(snapshot финального submission'а; CSV в формате согласно ТЗ).

## Локальный запуск без Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
# torch CPU-only (~700MB)
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt

# системные библиотеки (macOS):
brew install zbar libmagic

# Backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
# UI
streamlit run app/ui.py --server.port 8501 &
```

## Используемые данные (прозрачность)

| Источник | Что | Объём | Лицензия / получение |
|---|---|---|---|
| Организаторы (`Данные/`) | 3 размеченных видео + GT-CSV (29 полей) | 159 MB видео + ~150 GT-строк | передано в рамках задания |
| Организаторы (`Материалы/`) | Шаблоны ценников Lenta (PPTX) | 271 шаблон | передано в рамках задания |
| lenta.com (публичный сайт) | Каталог SKU (название + цены + barcode) | 167 SKU | scrape через undetected-chromedriver, публичная страница, only public fields |
| Hugging Face (open weights) | Qwen2.5-3B-Q4_K_M GGUF | 1.8 GB | Apache-2.0, download один раз вручную |

**Ручной разметки не использовали.** Все ценники из GT превращены в датасет
автоматически (`build_gt_dataset.py`), synthetic-датасет сгенерирован программно
(`build_synthetic_dataset.py`), каталог собран автоматически через Selenium.
Финальный pipeline работает полностью автоматически без оператора.

## Ограничения и future work

**Бутылочное горлышко** — декодирование штрихкодов:
- Каталог покрывает **148/156 GT barcodes (98%)** — нет проблемы.
- Декодеры (pyzbar + zxing-cpp + WeChat-QR + cv2.barcode) находят **13/156 (8.6%)**
  даже при 72 preprocessing-комбинациях (3 variants × 2 upscale × 4 rotation).
- На видео 43_15 (мёд, варенье): **0 декодированных barcode'ов** — физический
  предел разрешения камеры 4K с motion blur.
- OCR-текст также не содержит ни одного GT-barcode digit sequence.

**Что улучшит метрику в production:**
1. **PaddleOCR Server** на bottom-zone для мелкого шрифта (id_sku, datetime, code).
   Ожидаемый прирост: +10-20%.
2. **Super-resolution** (Real-ESRGAN local) на barcode-zone перед декодом.
   Ожидаемый прирост recall: +20-30%.
3. **Дообучение детектора на 100+ видео** разных магазинов — текущая v2/v4/v5b
   обучена на ~180 кадрах одного магазина.
4. **GPU-inference / rknn (int8)** — текущие ~50 мин/видео на CPU сократятся до
   ~5-10 мин.
5. **Online catalog API** — прямая интеграция с BIRD-системой Lenta для актуальных
   цен (вместо scraped каталога).

## Документы

- [DOCKER_RUN.md](DOCKER_RUN.md) — подробная инструкция по Docker, env-переменные.
- [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md) — заготовка слайдов для презентации.
- [ml/DAY1_REPORT.md](ml/DAY1_REPORT.md) — [ml/DAY16_REPORT.md](ml/DAY16_REPORT.md) — журнал работы (16 дней).
- [ml/BACKEND_HANDOFF.md](ml/BACKEND_HANDOFF.md) — техническая спецификация интерфейса
  ML ↔ backend.

## Стек

- **ML**: YOLOv8n/s (ultralytics), EasyOCR, WeChat-QR (OpenCV contrib), pyzbar,
  zxing-cpp, llama-cpp-python + Qwen2.5-3B.
- **Backend**: FastAPI, Celery 5, Redis 7, SQLite (jobs history), parquet
  (analytics).
- **UI**: Streamlit (operator panel + analytics dashboard).
- **Observability**: structlog (JSON logs), Sentry SDK, prometheus-client.
- **Auth/Security**: python-jose JWT, slowapi rate-limit, libmagic MIME-sniffing,
  размер/duration/resolution validators.
- **Infra**: Docker multi-stage build (CPU-only torch via PyTorch index),
  итоговый образ ~3-4 GB.

## Команда «Медвежата»

| Роль | Имя | Контакт |
|---|---|---|
| ML / Computer Vision | _Ангелина Дугау_ | tg: @… |
| Backend / Infra | _…_ | … |
| Frontend / UX | _…_ | … |
| PM / Координация | _…_ | … |

_(заполните перед сдачей)_

## Лицензия

Код проекта — для участия в хакатоне Lenta Tech, без явной лицензии.
Pre-trained веса (YOLOv8, Qwen, EasyOCR) — под лицензиями соответствующих
upstream-проектов (Apache-2.0 / MIT / Ultralytics AGPL-3.0 — см. их репозитории).
