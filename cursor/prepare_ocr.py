# -*- coding: utf-8 -*-
"""Подготовка датасета для OCR."""

import pandas as pd
import cv2
from pathlib import Path
import json
from tqdm import tqdm

print("="*60)
print("Подготовка датасета для OCR")
print("="*60)

# Пути
video_dir = Path('Данные')
output_dir = Path('backend/ml/data/ocr_dataset')
images_dir = output_dir / 'images'
labels_dir = output_dir / 'labels'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Находим CSV
csv_files = list(video_dir.glob('**/*.csv'))
print(f"Найдено CSV: {len(csv_files)}")

total_images = 0

for csv_path in csv_files[:1]:  # Берем первое для теста
    video_name = csv_path.stem
    video_file = None

    for ext in ['.mp4', '.mov', '.avi']:
        candidate = csv_path.parent / f"{video_name}{ext}"
        if candidate.exists():
            video_file = candidate
            break

    if not video_file:
        print(f"[!] Видео не найдено: {video_name}")
        continue

    print(f"\n[+] Обработка: {video_file.name}")

    # Загружаем разметку
    df = pd.read_csv(csv_path)

    # Преобразуем координаты
    for col in ['x_min', 'y_min', 'x_max', 'y_max']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Открываем видео
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Группируем по времени
    for timestamp, group in tqdm(df.groupby('frame_timestamp'), desc="Кадры"):
        frame_num = int((timestamp / 1000) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Обрабатываем ценники
        for idx, row in group.iterrows():
            x_min, y_min = int(row['x_min']), int(row['y_min'])
            x_max, y_max = int(row['x_max']), int(row['y_max'])

            h, w = frame.shape[:2]
            x_min, x_max = max(0, min(x_min, w)), max(0, min(x_max, w))
            y_min, y_max = max(0, min(y_min, h)), max(0, min(y_max, h))

            if x_max <= x_min or y_max <= y_min:
                continue

            # Вырезаем ценник
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                continue

            # Проверяем качество
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F).var()

            if laplacian < 30:  # Слишком размыто
                continue

            # Сохраняем
            img_name = f"{video_name}_{int(timestamp)}_{idx}.jpg"
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Метаданные
            label_data = {
                'image': img_name,
                'product_name': str(row.get('product_name', '')).strip(),
                'price_default': str(row.get('price_default', '')).strip(),
                'price_card': str(row.get('price_card', '')).strip(),
                'barcode': str(row.get('barcode', '')).strip(),
                'id_sku': str(row.get('id_sku', '')).strip(),
                'quality': float(laplacian)
            }

            label_path = labels_dir / f"{img_name.replace('.jpg', '.json')}"
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False)

            total_images += 1

    cap.release()
    print(f"  Извлечено: {total_images} изображений")

print(f"\n{'='*60}")
print(f"Всего изображений: {total_images}")
print(f"Сохранено в: {output_dir}")
print(f"{'='*60}")
