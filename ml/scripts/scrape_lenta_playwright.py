"""Playwright-скрейпер каталога Lenta (обходит Qrator/Cloudflare).

Открывает реальный браузер (Chromium), Qrator выполняет JS-челлендж,
сайт загружается. Берём product URLs из каталога, потом с каждого
парсим JSON-LD.

Установка:
    pip install playwright
    playwright install chromium

Использование:
    python3 ml/scripts/scrape_lenta_playwright.py --max 1500 \\
        --out ml/data/lenta_catalog.parquet
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

CATEGORIES = [
    # из видео хакатона — приоритет
    "alkogol-17036",
    "molochnye-produkty-3",
    "napitki-4",
    "sladosti-1028",
    "konservaciya-94",
    "maslo-sousy-specii-20824",
    # дополнительно для покрытия
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
]


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
                "barcode": str(it.get("gtin13") or it.get("gtin")
                                or it.get("sku") or ""),
                "price": offers.get("price") or "",
                "url": it.get("url") or "",
                "brand": (it.get("brand", {}) or {}).get("name", "")
                          if isinstance(it.get("brand"), dict)
                          else str(it.get("brand") or ""),
                "category": it.get("category", ""),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog.parquet"))
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--pages_per_cat", type=int, default=5,
                    help="скан N страниц на категорию")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--show", action="store_true",
                    help="показывать браузер (для отладки)")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Установи: pip install playwright && playwright install chromium")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headless = not args.show

    with sync_playwright() as p:
        # Порядок: WebKit (нативно на macOS, не крашится на arm64) →
        # Firefox → Chromium (часто BUS_ADRALN на Apple Silicon).
        browser = None
        last_error = None
        for engine_name in ("webkit", "firefox", "chromium"):
            try:
                engine = getattr(p, engine_name)
                browser = engine.launch(headless=headless)
                print(f"  using engine: {engine_name}")
                break
            except Exception as e:
                last_error = e
                print(f"  {engine_name} failed: {e}")
        if browser is None:
            raise RuntimeError(f"no browser engine works: {last_error}")
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36",
            locale="ru-RU",
            viewport={"width": 1366, "height": 900},
        )
        page = ctx.new_page()

        # 1) разогрев — заходим на главную, Qrator пропускает
        print("Warm-up: opening lenta.com ...")
        page.goto("https://lenta.com/", wait_until="domcontentloaded",
                   timeout=60_000)
        time.sleep(3)  # дать Qrator пройти челлендж

        # 2) собираем product URLs из каждой категории.
        # Lenta — Angular SPA, нужно дождаться рендера + scroll для lazy load.
        all_urls: list[str] = []
        for cat in args.categories:
            for pg in range(1, args.pages_per_cat + 1):
                url = f"https://lenta.com/catalog/{cat}/?page={pg}"
                try:
                    page.goto(url, wait_until="domcontentloaded",
                               timeout=60_000)
                    # ждём появления карточек или networkidle (5 сек max)
                    try:
                        page.wait_for_selector(
                            'a[automation-id="productCard"], a[href*="/product/"]',
                            timeout=10_000)
                    except Exception:
                        pass
                    # скроллим вниз чтобы запустить lazy-load оставшихся карточек
                    for _ in range(3):
                        page.evaluate(
                            "window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(800)
                    # вытаскиваем href из DOM (не из raw HTML)
                    new = page.eval_on_selector_all(
                        'a[href*="/product/"]',
                        "els => Array.from(new Set(els.map(e => e.href)))")
                except Exception as e:
                    print(f"  {cat} p{pg}: error {e}")
                    break
                if not new:
                    break
                all_urls.extend(new)
            print(f"  {cat}: total URLs so far = {len(set(all_urls))}")
            if len(set(all_urls)) >= args.max:
                break
        all_urls = list(dict.fromkeys(all_urls))

        all_urls = list(dict.fromkeys(all_urls))[: args.max]
        if not all_urls:
            print("Не нашли product URL — структура поменялась.")
            browser.close()
            return
        print(f"\nUnique product URLs: {len(all_urls)}")

        # 3) JSON-LD с каждой страницы.
        # Fallback: если JSON-LD не отрендерился, выдираем напрямую из DOM
        # (название, цена, штрихкод обычно подгружаются через API после рендера).
        rows: list[dict] = []
        for u in tqdm(all_urls):
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=45_000)
                # ждём JSON-LD или main-price (что появится первым)
                try:
                    page.wait_for_function(
                        "document.querySelector('script[type=\"application/ld+json\"]')"
                        " || document.querySelector('.main-price')",
                        timeout=8_000)
                except Exception:
                    pass
                html = page.content()
                items = _parse_jsonld(html)
                if not items:
                    # fallback — собираем из DOM
                    info = page.evaluate("""() => {
                        const get = (sel) => document.querySelector(sel)?.textContent?.trim() || '';
                        const name = get('h1, [automation-id="product-page-name"], .product-card_name-block');
                        const price = get('.main-price');
                        const old_price = get('.old-price-product span');
                        // штрихкод обычно в характеристиках или скрытых атрибутах
                        let barcode = '';
                        const allText = document.body.innerText;
                        const m = allText.match(/(?:Штрихкод|EAN|Barcode)\\D{0,5}(\\d{12,13})/i);
                        if (m) barcode = m[1];
                        return { name, price, old_price, barcode };
                    }""")
                    if info and info.get("name"):
                        items = [{
                            "name": info["name"],
                            "barcode": info.get("barcode", ""),
                            "price": info.get("price", "").replace("\xa0", "").replace("₽", "").strip(),
                            "url": u,
                            "brand": "",
                            "category": "",
                        }]
                for it in items:
                    it.setdefault("url", u)
                    rows.append(it)
            except Exception:
                continue
        browser.close()

    if not rows:
        print("Страницы открылись, JSON-LD не нашли.")
        return

    df = pd.DataFrame(rows)
    df["barcode"] = df["barcode"].astype(str).str.replace(r"\D", "", regex=True)
    df = df.drop_duplicates(subset=["barcode", "name"], keep="first")
    out_path = Path(args.out)
    try:
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False, encoding="utf-8")
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} SKUs → {out_path}")
    with_bc = (df["barcode"].astype(str).str.len() >= 12).sum()
    print(f"   с barcode (≥12 цифр): {with_bc}")


if __name__ == "__main__":
    main()
