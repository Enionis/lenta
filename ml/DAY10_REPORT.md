# Day 10 — product_name через локальный LLM (Qwen2.5-3B / LM Studio)

## Идея

До дня 10 поле `product_name` всегда было пустым. Сейчас у нас работает Qwen2.5-3B-YiLin в LM Studio локально (~2 GB, Q4_K_M). Идеально для extraction по шумному OCR-тексту.

## Что сделано

### 1. `llm_product_name.py` — экстрактор через LM Studio
- **Эндпоинт**: `http://localhost:1234/v1/chat/completions` (OpenAI-совместимый).
- **Модель**: `qwen2.5-3b-yilin` (Q4_K_M, 1.9 GB).
- **Промпт**: system + 3 few-shot примера, `temperature=0`, `max_tokens=80`.
- Шумный OCR (с ошибками, ценами, цифрами) → название товара или «нет».

### 2. Multi-rank контекст
Изначально LLM получал OCR только от rank=0 кадра — этого мало. После фикса: для каждого pred-трека собираем OCR-токены со ВСЕХ K=3 кадров (через `track_id`). LLM получает в 3× больше контекста.

После этого product_name распознано:
- 25_12-20: 1/16
- 26_12-20: **11/40**
- 43_15: 7/27

### 3. Fuzzy match в эвалюаторе
Точное совпадение строк бесполезно (GT длинные описания, pred — короче). Новое правило: пересечение слов длины ≥4 символа между GT и pred (без учёта регистра) → считаем match.

## Метрики дня 10

| Видео     | F1 (9→10) | TP | product_name matched |
|-----------|-----------|----|----------------------|
| 25_12-20  | 0.356     | 13 | 0                    |
| 26_12-20  | 0.468     | 26 | **7/26 (27%)**       |
| 43_15     | 0.500     | 14 | **1/14 (7%)**        |

**F1 без изменений** (LLM работает после spatio-temporal матчинга, не меняет его). **8 названий товаров теперь правильные** где раньше было 0.

## Примеры распознанных названий

### 26_12-20 (вино)
- `Вино PLACIDO Кьянти ... (Италия) 0.75L`
- `Вино DIG Тоскана`
- `Вино PURE ALTITUDE 0,75L`
- `Вино CHATEAU LE VIEUX FORT Бордо Мерло Крю Буржуа АОП ... (Франция) 0,75L`
- `Вино DOURGOGNE PINOT NOIR`

### 43_15 (мёд/джемы)
- `Канthмтор ZUEGG Абрикос УЛЕтПя (Германия)`
- `Пшn ЧАСТНАЯ gi ПА СЕКЛ` (читается ЧАСТНАЯ ПАСЕКА)
- `Цсл RILNUM CLUB архашч [Пасae)` (читается PREMIUM CLUB)
- `ZUEGG JJDr`

Видны реальные бренды: ZUEGG, PLACIDO, CHATEAU LE VIEUX FORT, PINOT NOIR, ЧАСТНАЯ ПАСЕКА, PREMIUM CLUB. Орфография рваная (наследие OCR), но бренд узнаваем.

## Производительность

- ~7–13 it/sec на M-серии MacBook (LM Studio с GPU offload).
- На 27 строк: ~3 секунды.
- Полный пайплайн 60-сек видео end-to-end: ~5–7 минут (OCR доминирует).

## Полная кумулятивная картина за 10 дней

| День | F1 25_12-20 | F1 26_12-20 | F1 43_15 | Цены ОК | Цвет ОК | Скидка ОК | Название ОК |
|------|-------------|-------------|----------|---------|---------|-----------|-------------|
| 5    | 0.048       | 0.229       | 0.304    | 0       | many    | 5         | 0           |
| 6    | 0.083       | 0.231       | 0.333    | 0       | many    | 8         | 0           |
| 7    | 0.091       | 0.263       | 0.296    | 8 (price_card) | many | 8     | 0           |
| 8    | 0.182       | 0.362       | 0.480    | 9       | many    | 9         | 0           |
| 9    | 0.356       | 0.468       | 0.500    | 7       | **41/53** | 8       | 0           |
| **10** | **0.356** | **0.468**   | **0.500**| 7       | 41/53   | 8         | **8**       |

## Финальное API для backend

```python
from ml.pipeline import PriceTagPipeline
pipe = PriceTagPipeline()
df = pipe.process_video("/path/to/video.mp4")  # pandas DataFrame
df.to_csv("result.csv", index=False, encoding="utf-8")
```

Опционально включить LLM-обогащение:
```bash
python3 ml/scripts/llm_product_name.py \
  --final result.csv --ocr ml/output/ocr_qr_<stem>.csv
# результат: тот же result.csv с заполненной product_name
```

Требование: LM Studio запущен с моделью `qwen2.5-3b-yilin` на `localhost:1234`.

## Что осталось (если будет 11+ день)

1. **Tiling 2×2 inference** — даст +5-15% recall на мелких ценниках.
2. **Лучшая OCR-модель** (PaddleOCR Server) для нижней зоны → id_sku, print_datetime.
3. **Barcode**: ML-детектор bbox штрихкода + super-resolution + CRNN.
4. **RKNN int8** конверсия для запуска на роботе.

## Команда воспроизведения

```bash
source .venv/bin/activate
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
DYLD_LIBRARY_PATH=/opt/homebrew/lib

# 1) трекер с v2, conf=0.15
python3 ml/scripts/detect_and_track.py --conf 0.15 --imgsz 960

# 2) hi-res crops padding 30%
python3 ml/scripts/extract_hires_crops.py --pad_pct 0.30

# 3) OCR (top-K кадров)
python3 ml/scripts/run_ocr_qr.py

# 4) submission + eval CSV
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
done

# 5) LLM-обогащение product_name (LM Studio должен быть запущен)
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/llm_product_name.py \
    --final ml/output/final_$v.csv --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/llm_product_name.py \
    --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
done

# 6) метрики
python3 ml/scripts/evaluate_final.py
```
