"""Дополнительный OCR-проход на нижней зоне ценника.

Цели:
  - id_sku (12 цифр, обычно с префиксом 270/370/470)
  - print_datetime (DD.MM.YYYY HH:MM)
  - code (NNNNNN - NNNNNN)
  - barcode-цифры (13 цифр, иногда печатается под штрихкодом)

Шаги:
  1. Открываем каждый hi-res crop.
  2. Берём нижние 30% (типичная зона мелкого текста на ценниках Lenta).
  3. CLAHE + grayscale + upscale ×5 (INTER_CUBIC).
  4. EasyOCR с allowlist "0123456789-:./ " — только цифры и разделители.
  5. Пишем токены в ocr_qr_<video>.csv → новое поле `bottom_extra_text`.

Использование:
    python3 ml/scripts/extra_bottom_ocr.py
    python3 ml/scripts/extra_bottom_ocr.py --files ml/output/ocr_qr_43_15.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    return _READER


def preprocess_bottom(img: np.ndarray, bottom_pct: float = 0.30,
                      upscale: int = 5) -> np.ndarray:
    """Берём низ + CLAHE + upscale."""
    h, w = img.shape[:2]
    y_start = int(h * (1 - bottom_pct))
    bottom = img[y_start:, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    if upscale > 1:
        eq = cv2.resize(eq, (eq.shape[1] * upscale, eq.shape[0] * upscale),
                        interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def ocr_extra(crop_path: Path, allowlist: str = "0123456789-:./ ") -> str:
    img = cv2.imread(str(crop_path))
    if img is None:
        return ""
    proc = preprocess_bottom(img)
    if proc.shape[0] < 20 or proc.shape[1] < 20:
        return ""
    try:
        results = get_reader().readtext(proc, detail=1, paragraph=False,
                                          allowlist=allowlist)
    except Exception:
        return ""
    tokens = [str(t) for (_, t, _) in results if t and t.strip()]
    return " ".join(tokens)


def process(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if "bottom_extra_text" not in df.columns:
        df["bottom_extra_text"] = ""
    # колонка с пред-вычисленным crop'ом (предпочитаем hires_crop)
    cols = df.columns.tolist()
    path_col = ("hires_crop" if "hires_crop" in cols
                else "crop_path" if "crop_path" in cols else None)
    if path_col is None:
        print(f"  no crop column in {csv_path}")
        return

    n_done = 0
    for idx in tqdm(df.index, desc=csv_path.stem):
        if str(df.at[idx, "bottom_extra_text"]).strip():
            continue  # уже обработано
        cp = df.at[idx, path_col]
        if not isinstance(cp, str):
            continue
        p = Path(cp)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            continue
        text = ocr_extra(p)
        df.at[idx, "bottom_extra_text"] = text
        n_done += 1
    df.to_csv(csv_path, index=False)
    has_text = (df["bottom_extra_text"].astype(str).str.len() > 0).sum()
    print(f"  {csv_path.name}: новых OCR={n_done}, с непустым текстом={has_text}/{len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()
    files = ([Path(p) for p in args.files] if args.files
             else sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv")))
    for f in files:
        print(f"\n=== {f.name} ===")
        process(f)


if __name__ == "__main__":
    main()
