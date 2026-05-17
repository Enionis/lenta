"""День 13: WeChat QR на полном 4K кадре + привязка к bbox трека.

WeChat — ML-детектор на основе CNN, лучше cv2.QRCodeDetector на сложных QR.
На 4K-кадре QR-код занимает ~150-200 px (высокого качества) — WeChat
обычно справляется.

Для каждого уникального frame_idx из ocr_qr csv:
  1. читаем оригинальный 4K-кадр;
  2. WeChat детектит ВСЕ QR-коды на нём;
  3. привязываем каждый к ближайшему по геометрии track'у;
  4. дополняем ocr_qr csv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [ROOT / "Данные", ROOT / "Данные" / "Unlabeled"]
WEIGHTS = ROOT / "ml" / "weights" / "wechat_qr"


def find_video(name: str) -> Path | None:
    for d in DATA_DIRS:
        for sub in [d] + list(d.glob("*")):
            if sub.is_dir():
                p = sub / name
                if p.exists():
                    return p
    return None


def in_or_near(track_box, code_box, margin: float = 1.0):
    tx1, ty1, tx2, ty2 = track_box
    cx1, cy1, cx2, cy2 = code_box
    # QR обычно ВЫШЕ ценника. Сильно расширяем track вверх.
    tw, th = tx2 - tx1, ty2 - ty1
    ext_box = (tx1 - tw * margin, ty1 - th * (margin + 0.5),
               tx2 + tw * margin, ty2 + th * 0.5)
    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    ex1, ey1, ex2, ey2 = ext_box
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def process(ocr_csv: Path):
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return
    for c in ("barcode_raw", "qr_raw"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
    video_name = df.iloc[0]["video"]
    stem = ocr_csv.stem.replace("ocr_qr_", "")
    video = find_video(video_name)
    if video is None:
        print(f"  video not found: {video_name}")
        return

    try:
        wechat = cv2.wechat_qrcode_WeChatQRCode(
            str(WEIGHTS / "detect.prototxt"),
            str(WEIGHTS / "detect.caffemodel"),
            str(WEIGHTS / "sr.prototxt"),
            str(WEIGHTS / "sr.caffemodel"))
    except Exception as e:
        print(f"  WeChat load failed: {e}")
        return

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    df["frame_idx_compute"] = (df["frame_ts_ms"] * fps / 1000).round().astype(int)
    unique_frames = sorted(df["frame_idx_compute"].unique())

    n_qr_new = 0
    n_frames_with_qr = 0
    for fi in tqdm(unique_frames, desc=stem):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            results, points = wechat.detectAndDecode(frame)
        except Exception:
            continue
        if not results:
            continue
        codes: list[tuple[str, tuple[float, float, float, float]]] = []
        for s, pts in zip(results, points):
            if not s:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            codes.append((s, (min(xs), min(ys), max(xs), max(ys))))
        if not codes:
            continue
        n_frames_with_qr += 1

        rows_for_frame = df[df["frame_idx_compute"] == fi]
        for _, r in rows_for_frame.iterrows():
            if str(df.at[r.name, "qr_raw"]).strip():
                continue
            tb = (r.x_min_orig, r.y_min_orig, r.x_max_orig, r.y_max_orig)
            for s, cb in codes:
                if in_or_near(tb, cb):
                    df.at[r.name, "qr_raw"] = s
                    n_qr_new += 1
                    break
    cap.release()

    if "frame_idx_compute" in df.columns:
        df = df.drop(columns=["frame_idx_compute"])
    df.to_csv(ocr_csv, index=False)
    print(f"  frames_with_qr={n_frames_with_qr}/{len(unique_frames)}  "
          f"decoded new QRs: {n_qr_new}")


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
