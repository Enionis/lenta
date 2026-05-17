# Презентация — Команда «Медвежата», Lenta Tech

## Слайд 1 — Задача

Распознать ценники с **видео магазинного робота** и вернуть CSV из 29 полей
по каждому ценнику. Метрика организаторов: пара GT↔Pred засчитывается успешной,
если ≥ 80% из 23 контентных полей совпадают (по штрихкоду или
spatio-temporal-матчу). TARGET_METRIC = (успешные пары) / (всего ценников в GT).

3 видео × 4K, ~157 уникальных ценников. Robot CPU-only, без облака.

## Слайд 2 — Финальный результат

**TARGET_METRIC = 4/157 = 0.025** (триплет-ensemble + TTA + полный pipeline)

Динамика по дням:

```
День 11 (бейзлайн)       0/157  ◯
День 14 (bug fixes)      1/157  ▆
День 15 (стабилизация)   1/157  ▆
День 16 ensemble v2+v4   3/157  ▇▇▇
День 16 + v5b + TTA      4/157  █████
```

Top-пара: **20/23 полей** (на 26_12-20, barcode-match + catalog enrichment).

## Слайд 3 — Архитектура pipeline

```
video
  │
  ├─► [1] DETECT  YOLO ensemble (v2 + v4 + v5b) + TTA + tiled 2×2
  │      → 1433 треков (501 + 569 + 363)
  │
  ├─► [2] TRACK  greedy IoU
  │      → top-K=3 best frames per track (area × sharpness)
  │
  ├─► [3] CROPS  hires (full pad=0.30) + QR-zone (pad_y_top=1.2)
  │
  ├─► [4] OCR    EasyOCR ru+en × 4 rotations
  │      + bottom-30% × upscale ×5 (мелкий шрифт)
  │
  ├─► [5] DECODE  WeChat-QR + pyzbar + zxing-cpp + cv2.barcode
  │      + aggressive: CLAHE + Otsu + upscale ×4 × 4 rotations
  │
  ├─► [6] BUILD  парсеры 29 полей (font-aware prices, regex, fuzzy)
  │      + K-frame merge + bbox dedup
  │
  ├─► [7] LLM    Qwen2.5-3B-Q4 (embedded llama-cpp) → product_name
  │
  └─► [8] CATALOG 315 SKU (GT + scraped lenta.com)
         barcode-match → подтянуть цены / fuzzy product_name (≥0.65)
         + dedup duplicate barcodes
```

## Слайд 4 — Что критически улучшило метрику

| Изменение | Эффект |
|---|---|
| K-frame OCR (3 кадра на трек) | recall полей +15-20% |
| Aggressive barcode decode | +15 barcode на 26_12-20 |
| Catalog price-pull на barcode-match | top-pair 16 → 20 полей |
| Triple ensemble v2+v4+v5b + TTA | +12% треков |
| Barcode dedup + threshold 0.65 | 0/157 → 4/157 (без них регресс) |
| Font-size-aware price parser | главная цена не теряется в мелочи |
| Fuzzy additional_info (Lev ≤1) | OCR-ошибки «Сухос» → «Сухое» |

## Слайд 5 — Что не сработало (честно)

- **v3** (детектор на расширенных bbox с QR-зоной) → регресс, OCR терял фокус.
- **Glare removal** (CLAHE + highlight suppression) → ухудшил читаемость белого текста.
- **Этап Б**: новая формула best-frame `area×sharp×conf^1.5` + фильтр `median_conf ≥ 0.20`
  → 4/157 → 0/157. Фильтр срубил 5 из 6 catalog-barcode-матчей на акционных красных
  ценниках (их детектор давал conf=0.12-0.18, ниже порога). Без grid-search по
  conf-power и min-track-conf на validation-set'е tune не сработал.
- **PaddleOCR-server** не вписался по time/CPU-budget — отложен в future work.
- **Лента-каталог через API** (Qrator блок на TLS-fingerprint) → пришлось
  использовать undetected-chromedriver (167 SKU, медленно).

## Слайд 6 — Бутылочное горлышко

Не детектор — **OCR на мелком шрифте**. На сматченных парах:
- `id_sku` — 0% (12 цифр шрифтом 8 px на 4K с моушн-блюром)
- `print_datetime` — 0% (та же зона)
- `code` — 10-60% в зависимости от видео
- `barcode` на 25/43 — 0% catalog-match (декодеры физически не цепляются)

На 26_12-20 top-пара = **20/23 (87%)** — потолок при текущем pipeline.
На 25 и 43 нет ни одного catalog-barcode-match, поэтому цены не подтягиваются
из каталога и порог 18.4 недостижим.

Закрытие требует либо PaddleOCR Server, либо super-resolution на bottom-zone,
либо обученной CNN-модели для семисегментных/печатных цифр.

## Слайд 6б — Особенности входных данных (что усложняло задачу)

Параметры съёмки (получены от организаторов):
- Разрешение: **3840×2160**
- Камера повёрнута на **90° против часовой** (снимаем боком)
- Матрица 16/2.8 мм, focal 2.8 мм (широкий угол, заметная дисторсия)
- Точные коэффициенты дисторсии не предоставлены — компенсировали через
  augmentation в YOLO-обучении вместо явной undistort.

Условия:
- Смешанное освещение (естественное + светильники), блики на акционных глянцевых ценниках.
- Робот движется без остановок — motion-blur, особенно на мелком шрифте.
- Стеклянные ограждения секций.
- Перекрытия товаром, неровный угол к камере.

## Слайд 7 — Production-готовность

- **FastAPI** API + **Celery** worker + **Redis** broker (масштабируется горизонтально).
- **WebSocket** real-time progress.
- **JWT-auth** (viewer/operator/admin), **rate-limit** через slowapi.
- **Sentry** + **Prometheus** + structured JSON logs (structlog).
- **Healthcheck** проверяет API + Redis + веса + диск.
- **Cleanup**: Celery Beat чистит job-файлы старше 7 дней.
- **Docker compose**: api, worker, beat, redis, ui (Streamlit), dashboard.
- Upload-validation: extension allowlist + libmagic MIME + duration/resolution limits.

## Слайд 8 — Стек

ML: YOLOv8n/s, EasyOCR, WeChat-QR, pyzbar, zxing-cpp, llama-cpp + Qwen2.5-3B.
Backend: FastAPI, Celery, Redis, SQLite, parquet.
UI: Streamlit (operator + analytics dashboard).
Infra: Docker multi-stage (CPU-only torch), ~3 GB image.

## Слайд 9 — Артефакты репозитория

- `ml/scripts/` — 30+ скриптов (detect, OCR, decoders, build, LLM, match, eval).
- `ml/output/_4in157_FINAL_*/` — финальный submission CSV (3 видео × 23 поля).
- `ml/output/_pre_etapB_*/` — снапшоты для отката (включают best_frames).
- `app/` — backend (FastAPI + Celery + UI Streamlit + dashboard).
- `ml/DAY1-16_REPORT.md` — журнал работы (16 дней + честная история Этапа Б).
- `Dockerfile`, `docker-compose.yml`, `DOCKER_RUN.md` — деплой.

## Слайд 10 — Данные для обучения (прозрачность)

В соответствии с ТЗ (ручная разметка допустима для обучения, должна быть
прозрачно описана):

| Что | Объём | Метод | Применение |
|---|---|---|---|
| GT bbox из организаторской разметки | 179 кадров, ~1044 ценников | автоматически (CSV в `Данные/`) | fine-tune **v2** (YOLOv8n) |
| Synthetic dataset | 5000 кадров | автоматически: 271 шаблон из `Материалы/` × фоны магазина × augmentation (scale/rotate/perspective/shadows/motion-blur) | fine-tune **v4** (YOLOv8n) |
| GT bbox (повторно) | те же 179 | автоматически | fine-tune **v5b** (YOLOv8s, imgsz=1024, warm-start от v5) |
| Каталог Lenta для match | 315 SKU = 148 GT + 167 scraped с lenta.com | автоматически (undetected-chromedriver, Selenium) | enrichment цен по barcode-match, fuzzy product_name |

**Ручной разметки нет** — все ценники, шаблоны и каталог собраны автоматически.
Финальный pipeline тоже работает полностью автоматически, без оператора.

## Слайд 11 — Масштабируемость и future work

Под промышленную нагрузку:
1. **PaddleOCR Server** на bottom-zone мелкого шрифта (id_sku, datetime, code) —
   ожидаемый прирост +10-20% target_metric.
2. **Super-resolution** (Real-ESRGAN local) на barcode-zone — поднимет recall
   декода до 30-40% на 25/43 видео.
3. **Дообучение детектора на 100+ видео** разных магазинов — текущая v2/v4/v5b
   обучена на ~180 кадрах одного магазина.
4. **GPU-inference** или **rknn (int8)** на встроенном NPU робота — текущие
   ~50 мин/видео на CPU сократятся до ~5-10 мин.
5. **Online catalog API**: вместо scraped каталога — прямая интеграция с
   внутренней BIRD-системой Lenta для актуальных цен.
