"""Диагностика: что показывает Lenta после загрузки через WebKit.

Сохраняет HTML страницы и скриншот → можно посмотреть, почему 0 URL.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "_probe"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    headless = "--show" not in sys.argv

    with sync_playwright() as p:
        browser = p.webkit.launch(headless=headless)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                       "Version/17.0 Safari/605.1.15",
            locale="ru-RU",
            viewport={"width": 1366, "height": 900},
        )
        page = ctx.new_page()

        for tag, url in [
            ("home", "https://lenta.com/"),
            ("cat_alkogol", "https://lenta.com/catalog/alkogol-17036/"),
            ("product_buket", "https://lenta.com/product/buket-n2-rossiya-1sht-664355/"),
        ]:
            print(f"\n--- {tag} ---")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            except Exception as e:
                print(f"  goto error: {e}")
                continue
            page.wait_for_timeout(4_000)  # дать Angular отрендериться
            html = page.content()
            (OUT / f"{tag}.html").write_text(html, encoding="utf-8")
            page.screenshot(path=str(OUT / f"{tag}.png"), full_page=False)
            print(f"  size = {len(html)} bytes")
            print(f"  saved: {OUT / f'{tag}.html'}  +  {tag}.png")
            # сколько ссылок на /product/ видно
            links = page.eval_on_selector_all(
                'a[href*="/product/"]',
                "els => Array.from(new Set(els.map(e => e.href)))")
            print(f"  product links found: {len(links)}")
            if links[:3]:
                print(f"    samples: {links[:3]}")
            # проверим селект магазина
            shop = page.evaluate(
                "document.querySelector('[automation-id=\"shop-selector\"]')?.textContent || ''")
            if shop:
                print(f"  shop selector text: {shop[:120]}")
            # title
            title = page.title()
            print(f"  page title: {title}")
            # есть ли productCard
            cards = page.eval_on_selector_all(
                '[automation-id="productCard"]',
                "els => els.length")
            print(f"  productCard count: {cards}")

        # вывод geo prompt
        print("\n--- Проверка геолокации/города ---")
        page.goto("https://lenta.com/", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        # сохраним финальный HTML главной — там может быть модалка
        (OUT / "home_after_wait.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(OUT / "home_after_wait.png"))
        print(f"  saved: {OUT / 'home_after_wait.html'}")
        browser.close()


if __name__ == "__main__":
    main()
