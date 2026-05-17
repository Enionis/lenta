"""
День 1: Анализ CSV разметки и подготовка данных.

Этот скрипт анализирует CSV файлы с разметкой ценников,
выводит статистику и готовит данные для обучения YOLO.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import Counter

def analyze_csv_files(data_dir: Path):
    """Анализ всех CSV файлов с разметкой."""

    csv_files = list(data_dir.glob("**/*.csv"))
    print(f"Найдено CSV файлов: {len(csv_files)}")
    print("=" * 80)

    all_data = []

    for csv_path in csv_files:
        print(f"\nАнализ: {csv_path}")
        print("-" * 40)

        df = pd.read_csv(csv_path)
        all_data.append(df)

        # Базовая статистика
        print(f"Количество ценников: {len(df)}")
        print(f"Колонок: {len(df.columns)}")

        # Проверка обязательных полей
        required_fields = [
            'filename', 'product_name', 'price_default', 'barcode',
            'frame_timestamp', 'x_min', 'y_min', 'x_max', 'y_max'
        ]

        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            print(f"⚠️  Отсутствуют поля: {missing_fields}")

        # Статистика по заполненности полей
        print("\nЗаполненность полей:")
        for col in df.columns:
            non_empty = df[col].notna().sum()
            pct = non_empty / len(df) * 100
            if pct < 100:
                print(f"  {col}: {non_empty}/{len(df)} ({pct:.1f}%)")

        # Уникальные видео
        videos = df['filename'].unique()
        print(f"\nУникальные видео: {videos}")

        # Диапазон времени
        if 'frame_timestamp' in df.columns:
            ts_min = df['frame_timestamp'].min()
            ts_max = df['frame_timestamp'].max()
            print(f"\nДиапазон времени: {ts_min} - {ts_max} ms")
            print(f"Длительность: {(ts_max - ts_min) / 1000:.1f} сек")

        # Анализ цен
        if 'price_default' in df.columns:
            try:
                prices = df['price_default'].replace('нет', np.nan)
                prices = prices.str.replace(',', '.', regex=False)
                prices = pd.to_numeric(prices, errors='coerce')
                print(f"\nЦены: min={prices.min():.2f}, max={prices.max():.2f}, avg={prices.mean():.2f}")
            except:
                pass

        # Цвета ценников
        if 'color' in df.columns:
            colors = df['color'].value_counts()
            print(f"\nЦвета ценников:")
            for color, count in colors.items():
                print(f"  {color}: {count}")

        # Типы выкладки
        if 'special_symbols' in df.columns:
            symbols = df['special_symbols'].value_counts()
            print(f"\nТипы выкладки:")
            for sym, count in symbols.items():
                print(f"  {sym}: {count}")

        # Размеры bbox (преобразуем строки с запятой в числа)
        if all(c in df.columns for c in ['x_min', 'y_min', 'x_max', 'y_max']):
            def parse_num(val):
                if pd.isna(val):
                    return np.nan
                return float(str(val).replace(',', '.'))

            x_min = df['x_min'].apply(parse_num)
            x_max = df['x_max'].apply(parse_num)
            y_min = df['y_min'].apply(parse_num)
            y_max = df['y_max'].apply(parse_num)

            widths = x_max - x_min
            heights = y_max - y_min
            print(f"\nРазмеры bbox:")
            print(f"  Ширина: {widths.min():.0f} - {widths.max():.0f} px (avg: {widths.mean():.0f})")
            print(f"  Высота: {heights.min():.0f} - {heights.max():.0f} px (avg: {heights.mean():.0f})")

    # Объединенная статистика
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print("\n" + "=" * 80)
        print("ОБЩАЯ СТАТИСТИКА")
        print("=" * 80)
        print(f"Всего ценников: {len(combined)}")
        print(f"Уникальных товаров: {combined['product_name'].nunique()}")

        # Сохранение объединенной статистики
        stats = {
            'total_price_tags': len(combined),
            'unique_products': combined['product_name'].nunique(),
            'csv_files': len(csv_files),
            'columns': list(combined.columns),
            'color_distribution': combined['color'].value_counts().to_dict() if 'color' in combined.columns else {},
            'bbox_stats': {
                'avg_width': 100.0,  # Placeholder - will be calculated properly with video
                'avg_height': 150.0,
            }
        }

        stats_path = Path('backend/ml/data/annotation_stats.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nСтатистика сохранена: {stats_path}")

    return all_data


def prepare_yolo_dataset(data_dir: Path, output_dir: Path):
    """Подготовка датасета в формате YOLO для обучения детектора."""

    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАТАСЕТА YOLO")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'labels').mkdir(exist_ok=True)

    csv_files = list(data_dir.glob("**/*.csv"))

    all_annotations = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        video_name = csv_path.stem  # например, "43_15"

        print(f"\nОбработка: {video_name}")

        # Группировка по времени (кадрам)
        for timestamp, group in df.groupby('frame_timestamp'):
            # Имя кадра
            frame_name = f"{video_name}_{int(timestamp)}"

            annotations = []

            for _, row in group.iterrows():
                # Преобразование bbox в формат YOLO (нормализованные x_center, y_center, width, height)
                # Примечание: нужно знать размеры кадра, пока используем условные 3840x2160 (4K)
                img_width = 3840  # Предполагаемый размер, нужно получить из видео
                img_height = 2160

                # Преобразуем строки с запятой в числа
                x_min = float(str(row['x_min']).replace(',', '.'))
                y_min = float(str(row['y_min']).replace(',', '.'))
                x_max = float(str(row['x_max']).replace(',', '.'))
                y_max = float(str(row['y_max']).replace(',', '.'))

                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Класс 0 = price_tag
                annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if annotations:
                # Сохранение аннотации
                label_file = output_dir / 'labels' / f"{frame_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotations))

                all_annotations.append({
                    'frame_name': frame_name,
                    'video': video_name,
                    'timestamp': timestamp,
                    'num_tags': len(annotations)
                })

    # Сохранение информации о датасете
    dataset_info = pd.DataFrame(all_annotations)
    dataset_info.to_csv(output_dir / 'dataset_info.csv', index=False)

    print(f"\n[OK] Датасет подготовлен:")
    print(f"   Аннотаций: {len(all_annotations)}")
    print(f"   Всего ценников: {dataset_info['num_tags'].sum()}")

    # Создание data.yaml для YOLO
    yaml_content = f"""path: {output_dir.absolute()}
train: images
train_labels: labels
val: images
val_labels: labels

nc: 1
names: ['price_tag']

# Классы
categories:
  0: price_tag
"""

    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"   Конфиг: {output_dir / 'data.yaml'}")

    return output_dir


def main():
    data_dir = Path('Данные')
    output_dir = Path('backend/ml/data/processed')

    # Анализ разметки
    all_data = analyze_csv_files(data_dir)

    # Подготовка датасета (требует видео для извлечения кадров)
    # Без видео создаем только структуру аннотаций
    prepare_yolo_dataset(data_dir, output_dir)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)
    print("\nСледующие шаги:")
    print("1. Скачайте видеофайлы в папку Данные/")
    print("2. Запустите 02_extract_frames.py для извлечения кадров")
    print("3. Запустите 03_train_yolo.py для обучения детектора")


if __name__ == '__main__':
    main()
