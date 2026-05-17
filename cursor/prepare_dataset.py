# -*- coding: utf-8 -*-
import pandas as pd
import cv2
import json
from pathlib import Path
from tqdm import tqdm

print("="*60)
print("Подготовка датасета OCR")
print("="*60)

dataset_dir = Path("backend/ml/data/ocr_dataset")
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

video_dir = Path("Данные")
csv_files = list(video_dir.glob("**/*.csv"))
print(f"Найдено CSV: {len(csv_files)}")

total = 0

for csv_path in csv_files:
    video_name = csv_path.stem
    video_file = None
    for ext in [".mp4", ".mov"]:
        if (csv_path.parent / (video_name + ext)).exists():
            video_file = csv_path.parent / (video_name + ext)
            break
    
    if not video_file:
        continue
    
    print(f"\nОбработка: {video_file.name}")
    
    df = pd.read_csv(csv_path)
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for ts, group in tqdm(df.groupby("frame_timestamp"), desc=video_name[:10]):
        frame_num = int((ts / 1000) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        for idx, row in group.iterrows():
            x1, y1 = int(row["x_min"]), int(row["y_min"])
            x2, y2 = int(row["x_max"]), int(row["y_max"])
            h, w = frame.shape[:2]
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap < 30:
                continue
            
            roi_big = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2))
            
            img_name = f"{video_name}_{int(ts)}_{idx}.jpg"
            cv2.imwrite(str(images_dir / img_name), roi_big, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            label = {
                "image": img_name,
                "product_name": str(row.get("product_name", "")).strip(),
                "price_default": str(row.get("price_default", "")).strip(),
                "barcode": str(row.get("barcode", "")).strip(),
                "quality": float(lap)
            }
            
            with open(labels_dir / img_name.replace(".jpg", ".json"), "w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False)
            
            total += 1
    
    cap.release()

print(f"\nИзвлечено: {total} изображений")
print("="*60)