"""
Извлечение кадров из видео для обучения YOLO.
"""

import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json


def extract_training_frames(
    data_dir: Path = Path('Данные'),
    output_dir: Path = Path('backend/ml/data/processed'),
    margin_ms: int = 500
):
    """Извлечь кадры из видео по разметке CSV."""

    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob('**/*.csv'))

    stats = {'videos': 0, 'frames': 0, 'errors': []}

    for csv_path in csv_files:
        video_name = csv_path.stem
        video_file = None

        # Ищем видео
        for ext in ['.mp4', '.mov', '.avi', '.MOV']:
            candidate = csv_path.parent / f"{video_name}{ext}"
            if candidate.exists():
                video_file = candidate
                break

        if not video_file:
            print(f"[!] Видео не найдено: {video_name}")
            stats['errors'].append(f"Missing: {video_name}")
            continue

        print(f"\n[+] Обработка: {video_file.name}")

        df = pd.read_csv(csv_path)
        timestamps = df['frame_timestamp'].unique()
        print(f"    Кадров для извлечения: {len(timestamps)}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            stats['errors'].append(f"Cannot open: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for ts in tqdm(timestamps, desc=f"    {video_name}"):
            # Центр диапазона
            frame_num = int((ts / 1000) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Сохраняем кадр
            frame_name = f"{video_name}_{int(ts)}"
            frame_path = images_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Создаем аннотацию YOLO
            group = df[df['frame_timestamp'] == ts]
            labels = []

            for _, row in group.iterrows():
                h, w = frame.shape[:2]

                # Преобразуем строки с запятой в float
                x_min = float(str(row['x_min']).replace(',', '.'))
                y_min = float(str(row['y_min']).replace(',', '.'))
                x_max = float(str(row['x_max']).replace(',', '.'))
                y_max = float(str(row['y_max']).replace(',', '.'))

                # YOLO формат: x_center, y_center, width, height (нормализованные)
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if labels:
                label_path = labels_dir / f"{frame_name}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))

            stats['frames'] += 1

        cap.release()
        stats['videos'] += 1

    # Сохраняем статистику
    with open(output_dir / 'extraction_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"[OK] Видео обработано: {stats['videos']}")
    print(f"[OK] Кадров извлечено: {stats['frames']}")
    print(f"{'='*50}")

    return stats


def update_data_yaml():
    """Обновить data.yaml для YOLO."""

    yaml_content = """# Dataset for price tag detection
path: E:/brbrbr/lent4/backend/ml/data/processed
train: images
val: images

test: images

nc: 1
names: ['price_tag']
"""

    yaml_path = Path('backend/ml/data/processed/data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ Обновлен: {yaml_path}")


if __name__ == '__main__':
    print("="*50)
    print("Извлечение кадров для обучения")
    print("="*50)

    extract_training_frames()
    update_data_yaml()

    print("\nГотово! Запускай обучение:")
    print("python backend/ml/notebooks/03_train_yolo.py --epochs 100 --model n")
