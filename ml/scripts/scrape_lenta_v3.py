"""Скрейпер v3 — через cloudscraper (обход Qrator/Cloudflare).

Lenta использует Qrator, обычный requests получает 401. cloudscraper
эмулирует браузер и обходит JS-challenge.

Установка: pip install cloudscraper

Использование:
    python3 ml/scripts/scrape_lenta_v3.py --out ml/data/lenta_catalog.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"


def _parse_jsonld(html: str) -> list[dict]:
    out = []
    for m in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.+?)</script>',
            html, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("@type")
            types = [t] if isinstance(t, str) else (t or [])
            if "Product" not in types:
                continue
            offers = it.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            out.append({
                "name": it.get("name", ""),
                "barcode": it.get("gtin13") or it.get("gtin") or it.get("sku") or "",
                "price": offers.get("price") or "",
                "url": it.get("url") or "",
                "brand": (it.get("brand", {}) or {}).get("name", ""),
                "category": it.get("category", ""),
            })
    return out


def _collect_product_urls_from_listing(scraper, slug: str, pages: int = 10):
    urls = []
    for page in range(1, pages + 1):
        url = f"https://lenta.com/catalog/{slug}/?page={page}"
        try:
            r = scraper.get(url, timeout=30)
        except Exception:
            break
        if r.status_code != 200:
            break
        # Lenta пишет ссылки на товары как /product/<slug>-<id>/ или
        # /catalog/<cat>/<slug>-<id>/. Ловим всё что заканчивается -<digits>/.
        new = re.findall(
            r'href="(/(?:product|catalog/[\w\-/]+)/[\w\-]+\-\d+/?)"',
            r.text)
        if not new:
            break
        urls.extend("https://lenta.com" + u for u in new)
        time.sleep(0.5)
    return list(dict.fromkeys(urls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog.parquet"))
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--categories", nargs="*", default=[
        # самые приоритетные для наших видео (вино, молочка, мёд/джемы)
        "alkogol-17036",
        "molochnye-produkty-3",
        "napitki-4",
        "sladosti-1028",
        "konservaciya-94",
        "maslo-sousy-specii-20824",
        # дополнительно
        "kofe-chajj-kakao-242",
        "syry-2",
        "kolbasa-sosiski-754",
        "myaso-i-ptica-136",
        "ovoshchi-frukty-144",
        "makarony-krupy-muka-25",
        "hleb-i-vypechka-165",
        "ryba-ikra-moreprodukty-183",
        "zamorozka-77",
        "sneki-20195",
        "zdorovoe-pitanie-1879",
    ])
    args = ap.parse_args()

    try:
        import cloudscraper
    except ImportError:
        print("Установи: pip install cloudscraper")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False})

    # 1) проверка что обходим
    print("Probe: lenta.com main page...")
    r = scraper.get("https://lenta.com/", timeout=30)
    print(f"  status={r.status_code}  size={len(r.text)}")
    if r.status_code != 200:
        print("❌ cloudscraper не пробил Qrator. Нужен Playwright/Selenium.")
        return

    # 2) собираем URL'ы товаров
    all_urls: list[str] = []
    for cat in args.categories:
        print(f"\n=== {cat} ===")
        urls = _collect_product_urls_from_listing(scraper, cat, pages=15)
        print(f"  product URLs: {len(urls)}")
        all_urls.extend(urls)
    all_urls = list(dict.fromkeys(all_urls))[: args.max]
    if not all_urls:
        print("Не нашли product URL — поменялась структура страницы.")
        return

    # 3) JSON-LD с каждой страницы
    print(f"\nFetching JSON-LD from {len(all_urls)} products ...")
    rows: list[dict] = []
    for u in tqdm(all_urls):
        try:
            r = scraper.get(u, timeout=30)
            if r.status_code != 200:
                continue
            for it in _parse_jsonld(r.text):
                it.setdefault("url", u)
                rows.append(it)
        except Exception:
            continue
        time.sleep(0.2)

    if not rows:
        print("Не извлекли JSON-LD — Lenta могла поменять разметку.")
        return

    df = pd.DataFrame(rows)
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].astype(str).str.replace(r"\D", "", regex=True)
    df = df.drop_duplicates(subset=["barcode", "name"], keep="first")
    out_path = Path(args.out)
    try:
        df.to_parquet(out_path, index=False) if out_path.suffix == ".parquet" \
            else df.to_csv(out_path, index=False, encoding="utf-8")
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} SKUs → {out_path}")
    with_bc = (df["barcode"].astype(str).str.len() >= 12).sum()
    print(f"   с валидным barcode (≥12 цифр): {with_bc}")


if __name__ == "__main__":
    main()
