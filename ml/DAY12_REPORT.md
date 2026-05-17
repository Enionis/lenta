# Day 12 — Агрессивный декод QR + расширенный discount

## Что сделано

### 1. `aggressive_decode.py` — массированный декод QR/barcode на crop'ах

Перебор preprocessing × scale × rotation × decoder:
- **Preprocessing** (3 варианта): color, CLAHE-equalized, Otsu-binary.
- **Scale**: ×2 и ×4 (для тонких линий QR).
- **Rotation**: 0/90/180/270.
- **Decoders**: pyzbar, zxing-cpp, cv2.barcode/QRCodeDetector — каждый со своими сильными местами.

**Результат:** на 140 OCR-крипов (26_12-20) декодировано **8 QR с валидными данными** вида:
```
barcode=3770001048277&price1=2526.31&price2=2399.99&price4=2184.99&aP=
```
До этого было 0. На 25_12-20 и 43_15 — ноль (мелкие QR, motion blur).

После K-frame merge → **3 трека из 40** в final CSV получают валидные `qr_code_barcode`, `price1_qr`, `price2_qr`, `price4_qr`. По 4 поля × 3 трека = 12 правильных полей дополнительно.

### 2. `whole_frame_codes2.py` — попытка декода на полном 4K кадре

Логика: QR на 4K кадре крупнее (~150 px) чем в crop'е (~50 px), pyzbar должен лучше справляться. Реализация: для каждого уникального `frame_idx` открываем 4K, прогоняем pyzbar+zxing с 4 поворотами.

**Результат:** 0 новых декодов на всех видео. Объяснение — fisheye-искажение и motion blur ломают QR даже на 4K, агрессивные upscale + бинаризация в `aggressive_decode` уже выжимают максимум.

### 3. Расширенный `discount_amount` regex

OCR превращает минус в апостроф/тильду/подчерк/точку/запятую. Добавил все варианты:
```python
DISCOUNT_RX = re.compile(r"[\-−–—'~_,.`]?\s?([1-9]\d)\s*%")
DISCOUNT_BIG_RX = re.compile(r"(?<!\d)[\-−–—'~]\s?([1-9]\d)(?!\d)")
```

`DISCOUNT_BIG_RX` ловит крупный шрифт без `%` (типа `-50` на ценнике в карты).

## Метрики

```
=== 25_12-20 ===  avg=0.465  ≥80%: 0/13  TARGET=0/57 = 0.000
=== 26_12-20 ===  avg=0.462  ≥80%: 0/26  TARGET=0/71 = 0.000  
=== 43_15    ===  avg=0.463  ≥80%: 0/14  TARGET=0/29 = 0.000
=== ИТОГО    ===  TARGET = 0/157 = 0.000
```

Прирост avg per-pair: **0.46 → 0.46** (стабильно, малый рост от 3 QR-успехов).

## Почему 3 QR не двигают метрику

Чтобы один сматченный pair получил ≥80% (≥19/23 полей правильно), ему нужно правильное распознавание почти всех полей. Сейчас расклад:
- ~6 полей всегда правильно за счёт «нет» по умолчанию (price_discount, wholesale_*, action_*, price3_qr) → **6/23 = 26%** базы
- +color (для 90%+ сматченных) → **~7/23 = 30%**
- +product_name (LLM, ~20% где сматчилось) → **+0–1 поле в среднем**
- +discount_amount (~20%) → **+0–1**
- +1 цена (~10%) → **+0**

Получается **~8/23 = 35%** в среднем — далеко от 80% порога.

**Ключевые блокеры:**
1. **барkоды/QR декодируются только на 5% строк** — это 4 поля × 95% строк = **23 поля × 17%** не закрываются.
2. **id_sku, print_datetime** — 0%, нужен лучший OCR на нижней зоне.
3. **price_default/_card** — точно совпадает только 5–30%, ±1% толерантность не помогает потому что OCR режет цифры.

## Стратегические выводы

С текущим качеством видео **80% per-tag практически недостижим** без работающего QR-декодера. На 4 QR-поля приходится ~17% всего «потолка». Реалистичные шаги:
1. **Tile-OCR на барkодную зону** (день 14): попробовать PaddleOCR на чёрных линиях штрихкода — он лучше на цифровых текстах.
2. **Заведомо «слабые» поля делать «нет»** по типу ценника (день 13): если на этом типе нет действия акции, ставить action_* всегда «нет» — мы и так это делаем для большинства, но можно расширить на color-dependent.
3. **Recall через tiling** (день 13): сейчас на 25_12-20 Pred=16 при GT=57 → мы пропускаем 41 ценник = 41 «нолика» в знаменателе. Tiling может удвоить детект → **сильнее тянет TARGET_METRIC**, даже если per-pair не растёт.

Реалистичный потолок без работающего QR на этих видео — **target_metric 5–15%**, который достижим в основном за счёт идеально-сатчфайных треков с короткими product_name (типа `Мед ZUEGG`).

## Артефакты

- `ml/scripts/aggressive_decode.py` — мультидекодер
- `ml/scripts/whole_frame_codes2.py` — whole-frame pyzbar (бесполезен на этих видео)
- 8 QR-успехов в `ocr_qr_26_12-20.csv` (раньше 0)
- Обновлённый `parse_discount` с расширенной таблицей символов

## Команда воспроизведения

```bash
source .venv/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/aggressive_decode.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/whole_frame_codes2.py
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/llm_product_name.py \
    --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
done
python3 ml/scripts/evaluate_official.py
```
