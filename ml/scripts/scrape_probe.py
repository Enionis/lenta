"""Диагностика доступа к lenta.com — что отдаёт сайт.

Проверяет несколько URL и показывает: статус, размер, признаки JSON-LD,
блокировки CDN, форму CAPTCHA.
"""
from __future__ import annotations

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

PROBES = [
    "https://lenta.com/",
    "https://lenta.com/catalog/",
    "https://lenta.com/catalog/alcohol/",
    "https://lenta.com/sitemap.xml",
    "https://lenta.com/sitemap_products.xml",
    "https://lenta.com/api/v1/catalog/categories",
    "https://lenta.com/api/v2/stores/0001/skus",
    "https://lentaonline.com/",
    "https://lenta.com/robots.txt",
]


def main():
    sess = requests.Session()
    sess.headers.update(HEADERS)
    for url in PROBES:
        try:
            r = sess.get(url, timeout=15, allow_redirects=True)
            text = r.text or ""
            size = len(text)
            jsonld = "application/ld+json" in text
            captcha = ("captcha" in text.lower() or "challenge" in text.lower())
            content_type = r.headers.get("Content-Type", "")
            print(f"\n{url}")
            print(f"  status={r.status_code}  size={size}  "
                  f"content-type={content_type}")
            print(f"  JSON-LD: {jsonld}   captcha-like: {captcha}")
            # покажем первые 200 символов
            snippet = re.sub(r"\s+", " ", text)[:200]
            print(f"  body[:200]: {snippet}")
        except Exception as e:
            print(f"\n{url}\n  ERROR: {e}")


if __name__ == "__main__":
    import re
    main()
