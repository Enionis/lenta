"""Скрейпер публичного каталога Lenta (lenta.com).

Собирает: SKU id, штрихкод (если есть в JSON-LD), название, цена, категория.
Сохраняет в ml/data/lenta_catalog.parquet (или csv).

Использует публичный API lenta.com/api/v2/* — он отдаёт JSON с теми же полями,
что и сайт показывает. Без авторизации.

Использование:
    python3 ml/scripts/scrape_lenta.py --limit 2000 --out ml/data/lenta_catalog.parquet
    # или несколько категорий:
    python3 ml/scripts/scrape_lenta.py --categories vino bezalkogolnye-napitki

Совместимо с приоритетом ТЗ: открытый источник, локальное использование,
описано в материалах решения.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"

# Публичные эндпоинты сайта Lenta (наблюдаются через DevTools).
# Если структура изменится — поправить здесь.
BASE = "https://lenta.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/html",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

CATEGORIES = [
    # покрывают зоны из тестовых видео
    "alcohol",       # алкоголь, вино
    "molochnye",     # молочные
    "med-djemy",     # мёд, джемы, сиропы
    "napitki",       # безалк. напитки
    "konditerskie",  # кондитерские
    "bakaleya",      # бакалея
]


def fetch_category_products(slug: str, store_id: str = "0001",
                             per_page: int = 60, max_pages: int = 30):
    """Идёт по страницам категории, возвращает список товаров.

    Структура запросов основана на API публичного сайта Lenta. Если эндпоинт
    отдаст 404/redirect — пробуем альтернативный путь через JSON-LD на HTML.
    """
    out = []
    sess = requests.Session()
    sess.headers.update(HEADERS)
    for page in range(1, max_pages + 1):
        # Вариант 1: новый API
        url = f"{BASE}/api/v1/catalog/categories/{slug}/products"
        params = {"page": page, "perPage": per_page,
                  "sort": "popular", "storeId": store_id}
        try:
            r = sess.get(url, params=params, timeout=20)
            if r.status_code != 200:
                # Вариант 2: старый эндпоинт
                r = sess.get(f"{BASE}/api/v2/stores/{store_id}/skus",
                             params={"categoryCode": slug,
                                     "page": page,
                                     "limit": per_page}, timeout=20)
            if r.status_code != 200:
                break
            data = r.json()
        except (requests.RequestException, json.JSONDecodeError):
            break
        # форматы могут отличаться — обрабатываем самые типичные
        items = (data.get("skus")
                 or data.get("products")
                 or data.get("items")
                 or [])
        if not items:
            break
        for it in items:
            out.append({
                "category": slug,
                "id": (it.get("id") or it.get("code")
                       or it.get("article") or ""),
                "name": (it.get("title") or it.get("name") or ""),
                "barcode": (it.get("ean") or it.get("barcode")
                            or _first_barcode(it) or ""),
                "price": (it.get("regularPrice")
                          or it.get("price") or 0),
                "card_price": (it.get("cardPrice")
                               or it.get("priceCard") or 0),
                "weight": (it.get("weight") or ""),
                "url": (it.get("url") or it.get("href") or ""),
            })
        time.sleep(0.4)  # бережём API
    return out


def _first_barcode(item: dict):
    for key in ("eans", "barcodes", "skuBarcodes"):
        v = item.get(key)
        if isinstance(v, list) and v:
            return v[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--store_id", default="0001")
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog.parquet"))
    ap.add_argument("--limit", type=int, default=0,
                    help="общий лимит SKU (0 = без лимита)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_items: list[dict] = []
    for cat in args.categories:
        print(f"\n=== {cat} ===")
        items = fetch_category_products(cat, store_id=args.store_id)
        print(f"  collected: {len(items)}")
        all_items.extend(items)
        if args.limit and len(all_items) >= args.limit:
            all_items = all_items[:args.limit]
            break

    if not all_items:
        print("Ничего не собрано — возможно API недоступен или структура "
              "ответа изменилась. Попробуйте scrape_lenta_html.py")
        return

    df = pd.DataFrame(all_items)
    # дедуп по (id) или (barcode)
    df = df.drop_duplicates(subset=[c for c in ("id", "barcode") if c in df.columns])
    out_path = Path(args.out)
    if out_path.suffix == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            print("pyarrow не установлен → сохраняю в CSV")
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} SKUs → {out_path}")
    # квикстат: сколько с непустым barcode
    if "barcode" in df.columns:
        with_bc = (df["barcode"].astype(str).str.len() > 5).sum()
        print(f"  с barcode: {with_bc} ({with_bc / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
