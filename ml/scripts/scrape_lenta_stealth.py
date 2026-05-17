"""Скрейпер Lenta через playwright-stealth (обход Qrator-детекции).

playwright-stealth убирает automation-следы из браузера: webdriver-флаг,
window.chrome, permissions, plugins, languages — Qrator не различает
браузер от автоматизации.

Установка:
    pip install playwright-stealth

Использование:
    python3 ml/scripts/scrape_lenta_stealth.py --max 500 \\
        --out ml/data/lenta_catalog_stealth.parquet
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
    "alkogol-17036",
    "molochnye-produkty-3",
    "napitki-4",
    "sladosti-1028",
    "konservaciya-94",
    "maslo-sousy-specii-20824",
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
                "barcode": str(it.get("gtin13") or it.get("gtin") or ""),
                "price": offers.get("price") or "",
                "url": it.get("url") or "",
                "brand": (it.get("brand", {}) or {}).get("name", "")
                          if isinstance(it.get("brand"), dict) else "",
                "category": it.get("category", ""),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog_stealth.parquet"))
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
        from playwright_stealth import Stealth
    except ImportError:
        print("Установи: pip install playwright-stealth")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headless = not args.show
    stealth = Stealth()

    with sync_playwright() as p:
        # сначала WebKit (стабильный на arm64), потом Firefox
        browser = None
        for engine in ("webkit", "firefox", "chromium"):
            try:
                browser = getattr(p, engine).launch(headless=headless)
                print(f"  using engine: {engine}")
                break
            except Exception as e:
                print(f"  {engine} failed: {e}")
        if not browser:
            return
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                       "Version/17.0 Safari/605.1.15",
            locale="ru-RU",
            viewport={"width": 1366, "height": 900},
        )
        # ВКЛЮЧАЕМ STEALTH
        stealth.apply_stealth_sync(ctx)
        page = ctx.new_page()

        # warm-up
        print("Warm-up...")
        try:
            page.goto("https://lenta.com/", wait_until="domcontentloaded",
                       timeout=45_000)
            page.wait_for_timeout(5_000)
            title = page.title()
            print(f"  title: {title}")
            if "403" in title or "1188" in str(len(page.content())):
                print("❌ Stealth не помог — Qrator пробил защиту.")
                browser.close()
                return
        except Exception as e:
            print(f"  goto error: {e}")
            browser.close()
            return

        all_urls = []
        for cat in args.categories:
            try:
                page.goto(f"https://lenta.com/catalog/{cat}/",
                           wait_until="domcontentloaded", timeout=45_000)
                try:
                    page.wait_for_selector(
                        'a[href*="/product/"]', timeout=10_000)
                except Exception:
                    pass
                for _ in range(3):
                    page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(800)
                new = page.eval_on_selector_all(
                    'a[href*="/product/"]',
                    "els => Array.from(new Set(els.map(e => e.href)))")
            except Exception as e:
                print(f"  {cat}: {e}")
                continue
            all_urls.extend(new)
            print(f"  {cat}: +{len(new)} → total {len(set(all_urls))}")
            if len(set(all_urls)) >= args.max:
                break

        all_urls = list(dict.fromkeys(all_urls))[: args.max]
        if not all_urls:
            print("Не нашли product URL.")
            browser.close()
            return

        rows = []
        for u in tqdm(all_urls):
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=45_000)
                page.wait_for_timeout(1_000)
                html = page.content()
                items = _parse_jsonld(html)
                if not items:
                    info = page.evaluate("""() => {
                        const get = (s) => document.querySelector(s)?.textContent?.trim() || '';
                        const name = get('h1, [automation-id="product-page-name"]');
                        const price = get('.main-price');
                        const all = document.body.innerText;
                        const m = all.match(/(?:Штрихкод|EAN)\\D{0,5}(\\d{12,13})/i);
                        return { name, price, barcode: m ? m[1] : '' };
                    }""")
                    if info.get("name"):
                        items = [{
                            "name": info["name"],
                            "barcode": info.get("barcode", ""),
                            "price": str(info.get("price", "")).replace("\xa0", "").replace("₽", "").strip(),
                            "url": u, "brand": "", "category": "",
                        }]
                for it in items:
                    it.setdefault("url", u)
                    rows.append(it)
            except Exception:
                continue
        browser.close()

    if not rows:
        print("Пусто.")
        return
    df = pd.DataFrame(rows)
    df["barcode"] = df["barcode"].astype(str).str.replace(r"\D", "", regex=True)
    df = df.drop_duplicates(subset=["barcode", "name"], keep="first")
    out_path = Path(args.out)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ {len(df)} SKUs → {out_path}")
    with_bc = (df["barcode"].astype(str).str.len() >= 12).sum()
    print(f"   c barcode: {with_bc}")


if __name__ == "__main__":
    main()
