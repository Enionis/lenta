"""День 12: whole-frame декод с pyzbar/zxing на 4K кадре.

На crop'е QR ~50px высотой, на полном 4K кадре те же ~150-200px —
декодеры справляются заметно лучше. Делаем второй проход:
для каждого уникального frame_idx из ocr_qr csv открываем оригинальный
видеокадр, прогоняем pyzbar/zxing с upscale + 4 ориентациями.
Найденный код привязываем к ближайшему bbox.

Этот скрипт ДОПОЛНЯЕТ aggressive_decode: запускать оба, второй проход
заполнит то, что не нашли в crop'ах.
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

try:
    from pyzbar import pyzbar as _zbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False

try:
    import zxingcpp as _zxing
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False


def find_video(name: str) -> Path | None:
    for d in DATA_DIRS:
        for sub in [d] + list(d.glob("*")):
            if sub.is_dir():
                p = sub / name
                if p.exists():
                    return p
    return None


def in_or_near(track_box, code_box):
    tx1, ty1, tx2, ty2 = track_box
    cx1, cy1, cx2, cy2 = code_box
    ix1, iy1 = max(tx1, cx1), max(ty1, cy1)
    ix2, iy2 = min(tx2, cx2), min(ty2, cy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    code_area = (cx2 - cx1) * (cy2 - cy1)
    if code_area > 0 and inter / code_area >= 0.4:
        return True
    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    margin_x = (tx2 - tx1) * 0.25
    margin_y = (ty2 - ty1) * 0.25
    return (tx1 - margin_x <= cx <= tx2 + margin_x and
            ty1 - margin_y <= cy <= ty2 + margin_y)


def process(ocr_csv: Path):
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return
    # приводим к object чтобы pandas не cast'нул в float
    for c in ("barcode_raw", "qr_raw"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
    video_name = df.iloc[0]["video"]
    stem = ocr_csv.stem.replace("ocr_qr_", "")
    video = find_video(video_name)
    if video is None:
        print(f"  video not found: {video_name}")
        return

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df["frame_idx_compute"] = (df["frame_ts_ms"] * fps / 1000).round().astype(int)
    unique_frames = sorted(df["frame_idx_compute"].unique())

    n_bc_new = 0
    n_qr_new = 0

    def to_orig(box, rot):
        x1, y1, x2, y2 = box
        if rot == 0:
            return (x1, y1, x2, y2)
        if rot == 90:
            return (y1, W_frame - x2, y2, W_frame - x1)
        if rot == 180:
            return (W_frame - x2, H_frame - y2, W_frame - x1, H_frame - y1)
        if rot == 270:
            return (H_frame - y2, x1, H_frame - y1, x2)
        return (x1, y1, x2, y2)

    for fi in tqdm(unique_frames, desc=stem):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue

        codes_b: list[tuple[str, tuple[float, float, float, float]]] = []
        codes_q: list[tuple[str, tuple[float, float, float, float]]] = []

        # pyzbar на полном кадре + 4 поворота. Без upscale (4K и так большой).
        if HAS_PYZBAR:
            for rot in (0, 90, 180, 270):
                if rot == 0:
                    fr = frame
                elif rot == 90:
                    fr = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    fr = cv2.rotate(frame, cv2.ROTATE_180)
                else:
                    fr = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                try:
                    for r in _zbar.decode(fr):
                        if not r.data:
                            continue
                        s = r.data.decode("utf-8", errors="ignore")
                        x, y, w, h = r.rect.left, r.rect.top, r.rect.width, r.rect.height
                        bb = (x, y, x + w, y + h)
                        bb = to_orig(bb, rot)
                        if r.type == "QRCODE":
                            codes_q.append((s, bb))
                        else:
                            codes_b.append((s, bb))
                except Exception:
                    pass

        if not codes_b and not codes_q:
            continue

        rows_for_frame = df[df["frame_idx_compute"] == fi]
        for _, r in rows_for_frame.iterrows():
            tb = (r.x_min_orig, r.y_min_orig, r.x_max_orig, r.y_max_orig)
            if not str(df.at[r.name, "barcode_raw"]).strip():
                for s, cb in codes_b:
                    if in_or_near(tb, cb):
                        df.at[r.name, "barcode_raw"] = s
                        n_bc_new += 1
                        break
            if not str(df.at[r.name, "qr_raw"]).strip():
                for s, cb in codes_q:
                    if in_or_near(tb, cb):
                        df.at[r.name, "qr_raw"] = s
                        n_qr_new += 1
                        break
    cap.release()

    if "frame_idx_compute" in df.columns:
        df = df.drop(columns=["frame_idx_compute"])
    df.to_csv(ocr_csv, index=False)
    print(f"  decoded new: barcodes={n_bc_new}, QRs={n_qr_new}")


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
