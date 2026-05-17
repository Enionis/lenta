"""Скрейпер Lenta через API напрямую (без браузера).

API endpoint: https://lenta.com/api-gateway/v1/catalog/items/{id}
Возвращает JSON с name, prices, categories, slug, attributes.

ВАЖНО: Headers критичны — Qrator смотрит на их комбинацию.
Использует точные headers из HAR-файла реального браузера.

Использование:
    # тест на одном товаре
    python3 ml/scripts/scrape_lenta_api.py --test

    # массовый сбор: перебор ID в диапазоне
    python3 ml/scripts/scrape_lenta_api.py --start 600000 --end 700000 \\
        --max 2000 --out ml/data/lenta_catalog_api.parquet
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"

# Headers ключевы — без них Qrator блокирует.
# Взято из HAR-файла реального браузера. Device-ID можно генерировать.
def _headers():
    dev_id = str(uuid.uuid4())
    return {
        "Accept": "application/json",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/148.0.0.0 Safari/537.36"),
        "X-Platform": "omniweb",
        "X-Domain": "moscow",
        "X-Delivery-Mode": "pickup",
        "X-Device-OS": "Web",
        "X-Device-Web-Platform": "desktop_web",
        "X-Device-ID": dev_id,
        "DeviceID": dev_id,
        "X-Retail-Brand": "lo",
        "X-Organization-ID": "",
        "Client": "angular_web_0.0.2",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148"',
        "Referer": "https://lenta.com/",
    }


def fetch_item(sess: requests.Session, item_id: int):
    url = f"https://lenta.com/api-gateway/v1/catalog/items/{item_id}"
    params = {"timestamp": int(time.time() * 1000)}
    try:
        r = sess.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def to_row(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("display", {}).get("name") or ""
    if not name:
        return None
    prices = item.get("prices", {})
    cats = item.get("categories", [])
    main_cat = next((c.get("name") for c in cats if c.get("isMain")), "")
    return {
        "id": item.get("id", ""),
        "name": name,
        "slug": item.get("slug", ""),
        "barcode": "",  # API не отдаёт штрихкод публично
        "price": (prices.get("priceRegular", 0) or 0) / 100,
        "card_price": (prices.get("price", 0) or 0) / 100,
        "category": main_cat,
        "brand": next((a.get("value") for a in item.get("attributes", [])
                        if a.get("alias") == "brand"), ""),
        "url": f"https://lenta.com/product/{item.get('slug', '')}-{item.get('id')}/",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="дернуть один известный товар и показать результат")
    ap.add_argument("--start", type=int, default=600000)
    ap.add_argument("--end", type=int, default=700000)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog_api.parquet"))
    ap.add_argument("--delay", type=float, default=0.4,
                    help="задержка между запросами (сек)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(_headers())

    if args.test:
        for tid in (664355, 520575, 150538, 611106, 745022):
            item = fetch_item(sess, tid)
            if item is None:
                print(f"  {tid}: FAIL")
                continue
            row = to_row(item)
            print(f"  {tid}: {row['name'][:80]}  price={row['price']}  card={row['card_price']}")
        return

    rows = []
    found = 0
    failed_in_row = 0
    pbar = tqdm(range(args.start, args.end))
    for item_id in pbar:
        item = fetch_item(sess, item_id)
        if item is None:
            failed_in_row += 1
            if failed_in_row > 50:
                # возможно Qrator начал банить — увеличим задержку
                time.sleep(2)
                failed_in_row = 0
            time.sleep(args.delay)
            continue
        failed_in_row = 0
        row = to_row(item)
        if row:
            rows.append(row)
            found += 1
        pbar.set_postfix(found=found)
        if found >= args.max:
            break
        time.sleep(args.delay)

    if not rows:
        print("Ничего не нашли — API заблокирован или диапазон пустой.")
        return

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ {len(df)} SKUs → {out_path}")


if __name__ == "__main__":
    main()
