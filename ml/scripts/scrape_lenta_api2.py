"""Скрейпер Lenta API v2 — с SessionToken (правильный auth).

Шаги:
1. POST /api/rest/sessionGet → получаем SessionToken.
2. POST /api-gateway/v1/catalog/items/recommendations с seed item_id
   → 50 связанных товаров.
3. BFS: каждый новый товар → recommendations → ещё 50 связанных.
4. Дубли отбрасываем, накапливаем до --max.

Использование:
    # тест
    python3 ml/scripts/scrape_lenta_api2.py --test

    # массовый сбор
    python3 ml/scripts/scrape_lenta_api2.py --max 5000 \\
        --seeds 664355 520575 745022 \\
        --out ml/data/lenta_catalog_api2.parquet
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import deque
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"


def make_session(domain: str = "moscow") -> requests.Session:
    """Создаёт session с правильными заголовками и SessionToken."""
    sess = requests.Session()
    dev_id = str(uuid.uuid4())
    base_h = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.9",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/148.0.0.0 Safari/537.36"),
        "Origin": "https://lenta.com",
        "Referer": "https://lenta.com/",
        "X-Platform": "omniweb",
        "X-Domain": domain,
        "X-Delivery-Mode": "pickup",
        "X-Device-OS": "Web",
        "X-Device-Web-Platform": "desktop_web",
        "X-Device-ID": dev_id,
        "DeviceID": dev_id,
        "X-Retail-Brand": "lo",
        "Client": "angular_web_0.0.2",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148"',
    }
    sess.headers.update(base_h)

    # 1) sessionGet — получаем SessionToken
    body = {
        "Head": {
            "MarketingPartnerKey": "mp300-b1de0bac2c257f3257bf5ef2eea4ecbc",
            "Version": "web-12.0.587",
            "Client": "angular_web_0.0.2",
            "Method": "sessionGet",
            "SessionToken": "",
            "RequestId": f"sessionGet_{uuid.uuid4().hex[:14]}",
            "DeviceId": dev_id,
            "Domain": domain,
        },
        "Body": {},
    }
    r = sess.post(
        "https://lenta.com/api/rest/sessionGet",
        data={"request": json.dumps(body, ensure_ascii=False)},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"sessionGet failed: HTTP {r.status_code}\n{r.text[:200]}")
    data = r.json()
    token = data.get("Head", {}).get("SessionToken", "") or \
            data.get("Body", {}).get("SessionToken", "")
    if not token:
        raise RuntimeError(f"sessionGet returned no token: {data}")
    sess.headers["sessiontoken"] = token
    sess.headers["SessionToken"] = token
    print(f"  ✓ SessionToken получен: {token[:16]}...")
    return sess


def fetch_item(sess: requests.Session, item_id: int) -> dict | None:
    url = f"https://lenta.com/api-gateway/v1/catalog/items/{item_id}"
    params = {"timestamp": int(time.time() * 1000)}
    try:
        r = sess.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_recommendations(sess: requests.Session, seed_id: int,
                          limit: int = 50) -> list[dict]:
    """recommendType=2 — похожие товары."""
    url = "https://lenta.com/api-gateway/v1/catalog/items/recommendations"
    out = []
    for rt in ("2", "1", "3"):  # пробуем разные типы рекомендаций
        try:
            r = sess.post(
                url,
                json={"id": seed_id, "recommendType": rt,
                      "offset": 0, "limit": limit},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            # API может вернуть items в разных полях
            for key in ("items", "products", "skus", "list"):
                v = data.get(key) if isinstance(data, dict) else None
                if isinstance(v, list) and v:
                    out.extend(v)
                    break
            if not out and isinstance(data, list):
                out.extend(data)
        except Exception:
            continue
        if out:
            break
    return out


def to_row(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("display", {}).get("name") or ""
    if not name:
        return None
    prices = item.get("prices", {}) or {}
    cats = item.get("categories", []) or []
    main_cat = next((c.get("name") for c in cats if c.get("isMain")),
                    cats[0].get("name") if cats else "")
    return {
        "id": item.get("id", ""),
        "name": name,
        "slug": item.get("slug", ""),
        "barcode": "",  # API не отдаёт штрихкод
        "price": (prices.get("priceRegular", 0) or 0) / 100,
        "card_price": (prices.get("price", 0) or 0) / 100,
        "category": main_cat,
        "brand": next((a.get("value") for a in item.get("attributes", [])
                        if a.get("alias") == "brand"), ""),
        "url": f"https://lenta.com/product/{item.get('slug', '')}-{item.get('id')}/",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[664355, 520575, 745022, 611106, 750642,
                             349551, 562171, 901142, 150538])
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog_api2.parquet"))
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Создаём session с SessionToken...")
    try:
        sess = make_session()
    except Exception as e:
        print(f"❌ Не удалось получить SessionToken: {e}")
        return

    if args.test:
        print("\n--- test single fetch ---")
        for tid in args.seeds[:3]:
            item = fetch_item(sess, tid)
            if item is None:
                print(f"  {tid}: FAIL")
                continue
            row = to_row(item)
            print(f"  {tid}: {row['name'][:80]}")
        print("\n--- test recommendations ---")
        recs = fetch_recommendations(sess, args.seeds[0])
        print(f"  got {len(recs)} recs from seed {args.seeds[0]}")
        for r in recs[:5]:
            print(f"    id={r.get('id')}  name={r.get('name', '')[:80]}")
        return

    # BFS: seed → recommendations → next layer
    collected: dict[int, dict] = {}
    queue = deque(args.seeds)
    visited: set[int] = set()
    pbar = tqdm(total=args.max, desc="SKUs")
    while queue and len(collected) < args.max:
        seed_id = queue.popleft()
        if seed_id in visited:
            continue
        visited.add(seed_id)
        recs = fetch_recommendations(sess, seed_id)
        time.sleep(args.delay)
        new_this_round = 0
        for rec in recs:
            rid = rec.get("id")
            if not rid or rid in collected:
                continue
            row = to_row(rec)
            if row:
                collected[rid] = row
                new_this_round += 1
                pbar.update(1)
                if len(collected) >= args.max:
                    break
                # планируем BFS дальше
                if rid not in visited:
                    queue.append(rid)
        if new_this_round == 0:
            # пустой ответ — продолжаем
            pass
    pbar.close()

    if not collected:
        print("Ничего не собрали.")
        return
    df = pd.DataFrame(collected.values())
    out_path = Path(args.out)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ {len(df)} SKUs → {out_path}")


if __name__ == "__main__":
    main()
