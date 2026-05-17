"""Скрейпер каталога Lenta v2 — пробует несколько источников.

Источники (в порядке попытки):
1) lenta.com — JSON-LD на страницах категорий (parser HTML).
2) lentaonline.com — публичный JSON-API через cookies + xsrf.
3) lenta.com/sitemap.xml → product pages → JSON-LD.

Каждый product page содержит JSON-LD `<script type="application/ld+json">`
со схемой Product (Schema.org) с полями: name, gtin13 (это и есть EAN),
offers.price, image, brand. Это стабильный путь.

Использование:
    python3 ml/scripts/scrape_lenta_v2.py --max 2000 --out ml/data/lenta_catalog.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def _get(sess, url, **kw):
    """Безопасный GET с retry."""
    for _ in range(3):
        try:
            r = sess.get(url, timeout=30, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


def _parse_jsonld_product(html: str) -> list[dict]:
    """Ищем все JSON-LD блоки и фильтруем Product."""
    out = []
    for m in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.+?)</script>',
            html, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
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
                "image": it.get("image") or "",
                "brand": (it.get("brand", {}) or {}).get("name", ""),
                "category": it.get("category", ""),
            })
    return out


def _discover_product_urls_from_sitemap(sess) -> list[str]:
    """Пытаемся забрать sitemap.xml — там бывает список всех product URL."""
    urls: list[str] = []
    for sm in (
            "https://lenta.com/sitemap.xml",
            "https://lenta.com/sitemap_products.xml",
            "https://lenta.com/sitemap_index.xml",
            "https://lenta.com/sitemap1.xml",
    ):
        r = _get(sess, sm)
        if r is None:
            continue
        # вытаскиваем <loc>...</loc>
        for m in re.finditer(r"<loc>(.+?)</loc>", r.text):
            u = m.group(1).strip()
            if "/product/" in u or "/p/" in u or re.search(r"/\d{6,}$", u):
                urls.append(u)
            elif u.endswith(".xml") and "sitemap" in u:
                # вложенные sitemap
                r2 = _get(sess, u)
                if r2:
                    for m2 in re.finditer(r"<loc>(.+?)</loc>", r2.text):
                        u2 = m2.group(1).strip()
                        if "/product/" in u2 or "/p/" in u2:
                            urls.append(u2)
    return list(dict.fromkeys(urls))  # dedup сохраняя порядок


def _try_html_listing(sess, slug: str, pages: int = 5):
    """Получаем product-урлы с обычной HTML-страницы категории."""
    urls = []
    for page in range(1, pages + 1):
        url = f"https://lenta.com/catalog/{slug}/?page={page}"
        r = _get(sess, url)
        if r is None:
            return urls
        # ловим все ссылки на товары
        for m in re.finditer(r'href="(/product/[^"]+|/p/[^"]+)"', r.text):
            urls.append("https://lenta.com" + m.group(1))
        if not urls:
            break
    return list(dict.fromkeys(urls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog.parquet"))
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--categories", nargs="*", default=[
        "alcohol", "molochnye-produkty", "med-i-dzhemy",
        "napitki-bezalkogolnye", "konditerskie-izdeliya", "bakaleya",
        "vino", "pivo", "voda",
    ])
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # 1) пробуем sitemap
    print("Step 1: sitemap.xml ...")
    product_urls = _discover_product_urls_from_sitemap(sess)
    print(f"  found via sitemap: {len(product_urls)}")

    # 2) fallback: HTML-листинг по категориям
    if len(product_urls) < 100:
        print("Step 2: HTML category listings ...")
        for cat in args.categories:
            urls = _try_html_listing(sess, cat)
            print(f"  {cat}: +{len(urls)}")
            product_urls.extend(urls)
        product_urls = list(dict.fromkeys(product_urls))
        print(f"  total unique URLs: {len(product_urls)}")

    if not product_urls:
        print("\n❌ Не нашли ни одного товара. Возможно сайт за CDN/CAPTCHA.")
        print("   Попробуй опцию --probe чтобы посмотреть какой ответ возвращает сайт.")
        return

    # 3) Получаем JSON-LD с каждой страницы
    product_urls = product_urls[: args.max]
    print(f"\nStep 3: парсим JSON-LD с {len(product_urls)} страниц ...")
    rows: list[dict] = []
    for u in tqdm(product_urls):
        r = _get(sess, u)
        if r is None:
            continue
        items = _parse_jsonld_product(r.text)
        for it in items:
            it.setdefault("url", u)
            rows.append(it)
        time.sleep(0.15)  # бережём сайт

    if not rows:
        print("Страницы открываются, но JSON-LD не найден. Сайт мог изменить разметку.")
        return

    df = pd.DataFrame(rows)
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].astype(str).str.replace(r"\D", "", regex=True)
    df = df.drop_duplicates(subset=["barcode", "name"], keep="first")
    out_path = Path(args.out)
    if out_path.suffix == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n✅ Saved {len(df)} SKUs → {out_path}")
    with_bc = (df["barcode"].astype(str).str.len() >= 12).sum()
    print(f"   с валидным barcode (≥12 цифр): {with_bc} ({with_bc / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
