"""Обогащение final_<video>.csv через каталог Lenta.

Делает 2 вещи:
1) Если в pred есть barcode (распознанный) — looked up по нему в каталоге,
   подтягивает name, price, category. Это **повышает product_name accuracy**.
2) Если barcode нет, но есть product_name (от LLM / OCR) — fuzzy match по
   названию (rapidfuzz / difflib). Если score достаточно высокий — берём
   из каталога barcode и точный name.
   Это **косвенно даёт barcode** для матчинга к GT (поскольку barcode =
   primary key для оценки).

Использование:
    python3 ml/scripts/match_to_catalog.py \
        --final ml/output/final_43_15.csv \
        --catalog ml/data/lenta_catalog.parquet \
        --threshold 0.65 --overwrite
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# rapidfuzz быстрее, но если не установлен — fallback на difflib
try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False


def _normalize(s: str) -> str:
    """Канонизируем строку для матчинга: нижний регистр, убираем мусор."""
    s = str(s or "").lower()
    # убираем мусор от OCR (одиночные символы, скобки и т.д.)
    s = re.sub(r"[^\w\s\-.,/]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_barcode(s) -> str:
    """Только цифры, убираем .0 от pandas float-cast."""
    s = str(s or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return re.sub(r"\D", "", s)


def _build_index(catalog: pd.DataFrame):
    """Готовим:
       - barcode_idx: dict barcode → row idx
       - name_norm: list of normalized names
       - rows: catalog rows in order
    """
    cat = catalog.copy()
    if "barcode" not in cat.columns:
        cat["barcode"] = ""
    cat["_bc"] = cat["barcode"].map(_normalize_barcode)
    cat["_name_norm"] = cat["name"].map(_normalize)
    barcode_idx = {bc: i for i, bc in enumerate(cat["_bc"]) if len(bc) >= 12}
    return cat, barcode_idx


def _fuzzy_top1(query: str, names: list[str]) -> tuple[int, float]:
    """Вернёт (best_idx, score 0..1)."""
    if HAS_RAPIDFUZZ:
        m = rf_process.extractOne(query, names, scorer=fuzz.token_set_ratio)
        if m is None:
            return -1, 0.0
        # rapidfuzz возвращает (str, score, idx) либо (str, score)
        if len(m) >= 3:
            return int(m[2]), float(m[1]) / 100.0
        return names.index(m[0]), float(m[1]) / 100.0
    # fallback: difflib
    best_i, best_s = -1, 0.0
    for i, n in enumerate(names):
        s = SequenceMatcher(None, query, n).ratio()
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True,
                    help="final_<video>.csv (будет обновлён in-place)")
    ap.add_argument("--catalog", required=True,
                    help="каталог Lenta (.parquet или .csv)")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="мин. fuzzy score для матча по имени (0..1)")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписывать имя/штрихкод даже если они уже есть")
    args = ap.parse_args()

    final_path = Path(args.final)
    df = pd.read_csv(final_path)
    cat_path = Path(args.catalog)
    if cat_path.suffix == ".parquet":
        catalog = pd.read_parquet(cat_path)
    else:
        catalog = pd.read_csv(cat_path)
    print(f"Loaded final: {len(df)} rows, catalog: {len(catalog)} SKUs")

    # типизируем нужные колонки как STRING (не float!), иначе pandas
    # запишет barcode как "8.009621e+12" и evaluate_official не сматчит.
    # price_*/discount_amount — тоже приводим к object чтобы можно было
    # писать строки/значения из каталога.
    for c in ("barcode", "product_name", "price_default", "price_card",
               "price_discount", "discount_amount"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
    # явно превращаем barcode в строку с фиксированным форматом
    def _bc_str(x):
        s = str(x or "").strip()
        if s.endswith(".0"):
            s = s[:-2]
        # обработка научной нотации (например "8.009621e+12")
        if "e" in s.lower():
            try:
                f = float(s)
                s = f"{int(f)}"
            except Exception:
                pass
        return s
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].map(_bc_str)

    cat, bc_idx = _build_index(catalog)
    name_list = cat["_name_norm"].tolist()

    n_by_barcode = 0
    n_by_fuzzy = 0
    for idx, row in df.iterrows():
        # 1) Прямой match по barcode
        bc = _normalize_barcode(row.get("barcode"))
        if len(bc) >= 12 and bc in bc_idx:
            ci = bc_idx[bc]
            crow = cat.iloc[ci]
            if args.overwrite or not str(row.get("product_name") or "").strip():
                df.at[idx, "product_name"] = crow["name"]
            # подтягиваем цены из каталога ТОЛЬКО если по barcode (надёжный матч)
            # и только если pred пусто/«нет» или сильно отличается (>20%)
            for fld, cat_fld in (("price_default", "price"),
                                  ("price_card", "card_price")):
                if cat_fld not in crow.index:
                    continue
                cat_val = crow.get(cat_fld)
                try:
                    cat_f = float(str(cat_val).replace(",", "."))
                except Exception:
                    continue
                if cat_f <= 0:
                    continue
                cur = df.at[idx, fld] if fld in df.columns else ""
                cur_s = str(cur).strip()
                try:
                    cur_f = float(cur_s) if cur_s and cur_s.lower() != "nan" else None
                except Exception:
                    cur_f = None
                # перетираем если: пусто, NaN, или сильно отличается (>20% и int part не совпадает)
                if cur_f is None or (
                    abs(cur_f - cat_f) / max(cat_f, 1) > 0.20
                    and int(cur_f) != int(cat_f)
                ):
                    df.at[idx, fld] = f"{cat_f:.2f}"
            n_by_barcode += 1
            continue

        # 2) Fuzzy match по имени
        name = _normalize(row.get("product_name"))
        if not name or len(name) < 5:
            continue
        best_i, score = _fuzzy_top1(name, name_list)
        if best_i < 0 or score < args.threshold:
            continue
        crow = cat.iloc[best_i]
        if args.overwrite or not str(row.get("product_name") or "").strip():
            df.at[idx, "product_name"] = crow["name"]
        new_bc = _normalize_barcode(crow.get("barcode"))
        if len(new_bc) >= 12 and not bc:
            df.at[idx, "barcode"] = str(new_bc)  # str чтобы pandas не сделал float
            # подтягиваем цены если pred пустые
            for fld, cat_fld in (("price_default", "price"),
                                  ("price_card", "card_price")):
                if cat_fld not in crow.index:
                    continue
                try:
                    cat_f = float(str(crow.get(cat_fld)).replace(",", "."))
                except Exception:
                    continue
                if cat_f <= 0:
                    continue
                cur_s = str(df.at[idx, fld] if fld in df.columns else "").strip()
                if not cur_s or cur_s.lower() == "nan":
                    df.at[idx, fld] = f"{cat_f:.2f}"
            n_by_fuzzy += 1

    # ДЕДУП: если несколько pred-строк получили один и тот же barcode
    # (например, fuzzy подтянул на разные ценники один SKU), оставляем
    # одну — с самой большой bbox-площадью (более чёткий best frame).
    if "barcode" in df.columns:
        df["_bc_norm"] = df["barcode"].map(_bc_str)
        # площадь bbox для tie-breaker
        try:
            df["_area"] = (df["x_max"] - df["x_min"]).astype(float) \
                          * (df["y_max"] - df["y_min"]).astype(float)
        except Exception:
            df["_area"] = 0
        # для каждой группы по непустому EAN-13 — keep top-area
        valid = df["_bc_norm"].str.len().fillna(0).astype(int).ge(12)
        if valid.any():
            grp = df[valid].sort_values("_area", ascending=False)
            keep_idx = set(grp.drop_duplicates(subset=["_bc_norm"], keep="first").index)
            # на дублях стираем barcode + product_name (чтобы не было ложного match'а)
            dup_mask = valid & ~df.index.isin(keep_idx)
            n_dups = int(dup_mask.sum())
            if n_dups:
                df.loc[dup_mask, "barcode"] = ""
                # product_name оставляем (вдруг это другой реальный товар)
                print(f"  dedup: cleared barcode on {n_dups} duplicate rows")
        df = df.drop(columns=["_bc_norm", "_area"], errors="ignore")

    df.to_csv(final_path, index=False, encoding="utf-8")
    print(f"\nEnriched: barcode_match={n_by_barcode}, fuzzy_match={n_by_fuzzy}")
    print(f"Saved → {final_path}")


if __name__ == "__main__":
    main()
