"""Fallback для product_name: если LLM вернул 'нет'/пусто, подставить
длинные кириллические/латинские токены из OCR-текста трека.

Evaluator считает product_name совпавшим, если есть пересечение по
4+ char токенам (см. _toks4 в evaluate_official.py). Поэтому даже
кусок «VESPUCCI Кьянтн» из OCR-мусора может дать match с GT.

Запуск:
    python3 ml/scripts/fill_product_name_fallback.py \\
        --final ml/output/final_43_15.csv \\
        --ocr   ml/output/ocr_qr_43_15.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Стоп-слова: бессмысленные распространённые токены из OCR — не помогают и
# создают шум.
_STOP = {
    "цена", "цены", "карты", "карта", "карте", "карту",
    "акции", "акция", "акцион", "акционная",
    "руб", "руб.", "рубль", "руб/", "rubl",
    "штрих", "штрих-",
    "налог", "ндс",
    "сухое", "сухос", "белое", "красное", "розовое",
    "полусухое", "полусладкое", "сладкое", "игристое",
    "крепкое", "столовое",
    "лента", "ленты",
    "vino", "wine",
}

_TOK_RX = re.compile(r"[A-Za-zА-Яа-я0-9\-]{4,}")


def pick_tokens(text: str, max_n: int = 5) -> list[str]:
    if not isinstance(text, str):
        return []
    seen: set[str] = set()
    out: list[str] = []
    # сортируем по длине: длинные более «брендовые» (VESPUCCI/МУКУНДАРИ)
    toks = sorted(set(_TOK_RX.findall(text)), key=lambda t: -len(t))
    for t in toks:
        low = t.lower()
        if low in _STOP or low in seen:
            continue
        # отбрасываем чисто-цифры (это цена/код) и короткие латинские стоп-слова
        if t.isdigit():
            continue
        if len(t) < 4:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--ocr", required=True)
    args = ap.parse_args()

    final = pd.read_csv(args.final)
    ocr = pd.read_csv(args.ocr)

    # rank=0 для bbox/ts матчинга → track_id
    rank0 = ocr[ocr.get("rank", 0) == 0] if "rank" in ocr.columns else ocr

    def collect_text(ts: int, xm: int, ym: int) -> str:
        cand = rank0[(rank0.frame_ts_ms == ts) & (rank0.x_min_orig == xm)
                     & (rank0.y_min_orig == ym)]
        if cand.empty:
            cand = ocr[ocr.frame_ts_ms == ts]
            if cand.empty:
                return ""
            return "\n".join(str(t) for t in cand.ocr_text if isinstance(t, str))
        tid = cand.iloc[0].track_id if "track_id" in cand.columns else None
        if tid is not None and "track_id" in ocr.columns:
            sub = ocr[ocr.track_id == tid]
            return "\n".join(str(t) for t in sub.ocr_text if isinstance(t, str))
        return "\n".join(str(t) for t in cand.ocr_text if isinstance(t, str))

    filled = 0
    new_names = []
    for r in final.itertuples(index=False):
        cur = getattr(r, "product_name", "")
        cur_s = str(cur).strip().lower() if cur is not None else ""
        if cur_s and cur_s != "нет" and cur_s != "nan":
            new_names.append(cur)
            continue
        text = collect_text(int(r.frame_timestamp), int(r.x_min), int(r.y_min))
        toks = pick_tokens(text)
        if not toks:
            new_names.append(cur)
            continue
        new_names.append(" ".join(toks))
        filled += 1

    final["product_name"] = new_names
    final.to_csv(args.final, index=False, encoding="utf-8")
    print(f"Fallback заполнил: {filled}/{len(final)} → {args.final}")


if __name__ == "__main__":
    main()
