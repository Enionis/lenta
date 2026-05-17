# Day 14 — Первый ценник проходит порог 80%!

## TL;DR

**TARGET_METRIC = 1/157 = 0.006** (впервые > 0).

После 13 дней работы первый ценник набрал **19/23 = 82.6% ≥ 80%** и был засчитан официальной метрикой организаторов.

## Что сделано

### 1. Два критичных бага в `evaluate_official.py`
**Bug A: GT barcode имеет ".0" хвост** (pandas float-cast). Например `3298660031398.0`. Pred хранит как строку `3298660031398`. Сравнение через `re.sub(r"\D", "", s)` оставляло `32986600313980` против `3298660031398` → **точно совпавший QR помечался как ❌**.

Фикс — сначала убирать `.0`, потом digits:
```python
def _ean(x):
    s = _norm_str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return re.sub(r"\D", "", s)
```

То же для `id_sku` — добавил отдельную ветку с такой же нормализацией.

### 2. Bug в `parse_qr`: пустое значение → должно быть «нет»
QR-данные имеют формат `barcode=XXX&aP=` (значение `aP` пустое). До дня 14: `out["action_price_qr"] = ""` → в CSV NaN → ❌ в эвалюаторе (GT="нет"). Фикс: если значение пустое, возвращаем `"нет"`.

### 3. `barcode = qr_code_barcode` fallback
Если основной `parse_barcode` ничего не нашёл, но в QR декодирован EAN — это **тот же штрихкод на том же ценнике**. Используем QR-значение как fallback. Точно совпадает с GT.

### 4. Расширенный `parse_discount`
До дня 14 regex требовал минус перед `NN%`. Но OCR иногда даёт чистое `30%`. Добавил три варианта:
- `[−–—'~_,.\`]?\s?([1-9]\d)\s*%` — минус опционален
- `(?<!\d)([1-9]\d)\s*%` — без префикса
- `[−–—'~]\s?([1-9]\d)(?!\d)` — крупный шрифт без %

### 5. `parse_additional_info` с fuzzy-match
Ценник часто содержит «Сухое»/«Полусладкое»/«Игристое». OCR искажает: `"Сухое"` → `"Сухос"`. Внедрил **Левенштейн ≤ 1** для коротких слов (4-15 символов).

```python
ADD_INFO_KEYWORDS = {
    "сухое": "Сухое", "полусухое": "Полусухое",
    "сладкое": "Сладкое", "полусладкое": "Полусладкое",
    "красное": "Красное", "белое": "Белое",
    "розовое": "Розовое", "игристое": "Игристое",
}
```

### 6. `parse_special_symbols`
Ищем одиночные крупные кириллические буквы из ограниченного набора `[К, Ш, Ц]` — типичные «коды выкладки» на ценниках Ленты.

## Метрики

```
=== 25_12-20 ===  avg=0.431  ≥80%: 0/49  TARGET=0/57 = 0.000
=== 26_12-20 ===  avg=0.448  ≥80%: 1/57  TARGET=1/71 = 0.014  ← !!!
=== 43_15    ===  avg=0.461  ≥80%: 0/27  TARGET=0/29 = 0.000
=== ИТОГО    ===  TARGET = 1/157 = 0.006
```

### Топ-пары после фиксов

| Видео | Top scores из 23 | Лучшая пара |
|---|---|---|
| 26_12-20 | **19, 16, 15, 13, 13** | 1 ≥ 18.4 (80%) ✓ |
| 25_12-20 | 13, 12, 12, 12, 12 | до 80% не хватает 5+ полей |
| 43_15 | 12, 12, 12, 12, 11 | до 80% не хватает 7+ полей |

### Состав успешной пары (26_12-20, pi=64, gi=10): 19/23

✅ price_default (3157.89 ≈ 3157.0 через fuzzy), price_card (2299.99 ≈ 2299.0), price_discount (нет), discount_amount (-27%), color (red), qr_code_barcode (3298660031398), price1_qr (3157.89), price2_qr (2999.99), price3_qr (нет), price4_qr (2299.99), wholesale_* × 4 (нет), action_price_qr (нет), action_code_qr (нет), barcode (через fallback от qr_code_barcode) — **итого 14 ✓ полей**.

Плюс после дня 14 ещё **5 ✓** через фиксы (qr_code_barcode стал засчитываться, action_*/wholesale_* стало «нет» вместо NaN).

❌ Не закрытые: id_sku (770101701 vs 270101701901 — OCR недосчитал цифры), print_datetime (NaN), code (`026017 - 026015` — сложный паттерн), additional_info (`Сухое` — fuzzy не дотянулся), special_symbols (`Ш`).

## Почему остальные пары не дотягивают

Топ-пара 26_12-20 (16/23) теперь поднялась до 19. Следующая (15→16) поднялась до 16 — ей не хватает barcode (нет QR fallback потому что QR не декодирован) и других полей.

Главные узкие места теперь:
- **id_sku недотягивается на 1-3 цифры** — OCR режет крайние цифры. Нужно tile-OCR на ровно цифровую зону с upscale ×5.
- **print_datetime = 0** на всех — нет нижнего OCR-прохода с allowlist цифр и точек.
- **price_default/_card** на 25/43 видео сильно шумит — OCR не видит крупный шрифт правильно.
- **product_name** через LLM работает, но OCR на 25_12-20 даёт мало контекста (короткие токены).

## Стратегические выводы

**Главный урок дня:** часто **«не работающий»** результат — это просто **бag в учёте**, а не в распознавании. Два бага в эвалюаторе и один в QR-парсере дали `pred=value/GT=value` → `❌` без причины. После фикса — 19 полей сразу.

**Следующие шаги (день 15):**
1. Аккуратный tile-OCR на нижнюю **20%** crop'а с upscale ×5 и `allowlist="0123456789-:./"` для id_sku и print_datetime. Это даст +1-3 поля на топ-парах → ещё **2-3 ценника пройдут порог**.
2. Финальный grid-search по conf/padding/min_track_len.
3. RKNN int8 конверсия модели.
4. README, презентация.

## Артефакты

- Обновлены: `evaluate_official.py`, `build_final_csv.py`
- Новые парсеры: `parse_additional_info` (с Левенштейн), `parse_special_symbols`
- Эвалюатор теперь честно считает .0-хвостые EAN/SKU.

## Команда воспроизведения

```bash
source .venv/bin/activate

# полный pipeline на 1 видео
DYLD_LIBRARY_PATH=/opt/homebrew/lib
SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")

python3 ml/scripts/detect_and_track_tiled.py --conf 0.15 --imgsz 960
python3 ml/scripts/extract_hires_crops.py --pad_pct 0.30
python3 ml/scripts/extract_qr_crops.py
python3 ml/scripts/run_ocr_qr.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/decode_qr_crops.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/wechat_wholeframe.py
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 ml/scripts/aggressive_decode.py

for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/llm_product_name.py \
    --final ml/output/final_eval_$v.csv --ocr ml/output/ocr_qr_$v.csv
done
python3 ml/scripts/evaluate_official.py
```
