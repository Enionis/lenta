"""
День 1-2: Извлечение кадров из видео.

Этот скрипт:
1. Находит все видео файлы в папке Данные/
2. Извлекает ключевые кадры (где есть ценники согласно разметке)
3. Сохраняет кадры и обновляет аннотации YOLO
"""

import pandas as pd
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
import json


def extract_keyframes_from_videos(
    data_dir: Path,
    output_dir: Path,
    margin_ms: int = 500  # Добавка времени вокруг кадра
):
    """
    Извлечение ключевых кадров из видео.

    Args:
        data_dir: Папка с CSV разметкой и видео
        output_dir: Папка для сохранения кадров
        margin_ms: Добавка времени (±ms) для захвата движения
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Находим все CSV файлы
    csv_files = list(data_dir.glob("**/*.csv"))

    stats = {
        'videos_processed': 0,
        'frames_extracted': 0,
        'errors': []
    }

    for csv_path in csv_files:
        video_name = csv_path.stem  # например, "43_15"
        video_file = csv_path.parent / f"{video_name}.mp4"

        # Пробуем другие расширения
        if not video_file.exists():
            for ext in ['.mov', '.avi', '.MOV', '.AVI']:
                alt_path = csv_path.parent / f"{video_name}{ext}"
                if alt_path.exists():
                    video_file = alt_path
                    break

        if not video_file.exists():
            print(f"⚠️  Видео не найдено: {video_name}.*")
            stats['errors'].append(f"Video not found: {video_name}")
            continue

        print(f"\n🎥 Обработка: {video_file.name}")

        # Загружаем разметку
        df = pd.read_csv(csv_path)

        # Получаем уникальные временные метки
        timestamps = df['frame_timestamp'].unique()
        print(f"   Ключевых кадров: {len(timestamps)}")

        # Открываем видео
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            stats['errors'].append(f"Cannot open: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for ts in tqdm(timestamps, desc="   Кадры"):
            # Конвертируем ms в frame number
            frame_num = int((ts / 1000) * fps)

            # Диапазон кадров с учетом margin
            frame_start = max(0, int(((ts - margin_ms) / 1000) * fps))
            frame_end = min(total_frames, int(((ts + margin_ms) / 1000) * fps))

            # Берем средний кадр из диапазона
            best_frame_num = (frame_start + frame_end) // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Сохраняем кадр
            frame_name = f"{video_name}_{int(ts)}"
            frame_path = images_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            stats['frames_extracted'] += 1

        cap.release()
        stats['videos_processed'] += 1

    # Сохраняем статистику
    with open(output_dir / 'extraction_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Извлечение завершено:")
    print(f"   Видео обработано: {stats['videos_processed']}")
    print(f"   Кадров извлечено: {stats['frames_extracted']}")

    return output_dir


def verify_extraction(data_dir: Path, processed_dir: Path):
    """Проверка что все кадры извлечены корректно."""
    images_dir = processed_dir / 'images'

    csv_files = list(data_dir.glob("**/*.csv"))

    print("\n🔍 Проверка извлечения:")

    for csv_path in csv_files:
        video_name = csv_path.stem
        df = pd.read_csv(csv_path)

        expected_frames = len(df['frame_timestamp'].unique())
        extracted_frames = len(list(images_dir.glob(f"{video_name}_*.jpg")))

        status = "✅" if extracted_frames >= expected_frames * 0.9 else "⚠️"
        print(f"   {status} {video_name}: {extracted_frames}/{expected_frames} кадров")


def main():
    data_dir = Path('Данные')
    output_dir = Path('backend/ml/data/processed')

    # Проверяем наличие видео
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi', '*.MOV', '*.AVI']:
        video_files.extend(data_dir.glob(f"**/{ext}"))

    if not video_files:
        print("❌ Видео файлы не найдены!")
        print("   Пожалуйста, скачайте видео в папку Данные/")
        print("\n   Ожидаемые файлы:")

        for csv_path in data_dir.glob("**/*.csv"):
            video_name = csv_path.stem
            print(f"     - {csv_path.parent}/{video_name}.mp4")

        return

    print(f"✅ Найдено видео: {len(video_files)}")
    for vf in video_files:
        print(f"   - {vf}")

    # Извлечение кадров
    extract_keyframes_from_videos(data_dir, output_dir)

    # Проверка
    verify_extraction(data_dir, output_dir)

    print("\n" + "=" * 60)
    print("ГОТОВО! Кадры сохранены в:", output_dir / 'images')
    print("=" * 60)


if __name__ == '__main__':
    main()
