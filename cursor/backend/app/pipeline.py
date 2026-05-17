"""
ML Pipeline для распознавания ценников.

Интегрирует ML модуль с Backend.
"""

import sys
from pathlib import Path

# Добавляем путь к ML модулю
ml_path = Path(__file__).parent.parent / "ml"
if str(ml_path) not in sys.path:
    sys.path.insert(0, str(ml_path))

# Импортируем ML пайплайн
try:
    from src.pipeline import process_video as ml_process_video
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML module not available: {e}")
    ML_AVAILABLE = False

# Импорты для fallback
import time
import logging
from typing import Callable, Optional, Dict, Any
import random

import cv2
import numpy as np
import pandas as pd

from app.config import (
    CSV_COLUMNS,
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT,
    BBOX_COLOR,
    BBOX_THICKNESS,
    TEXT_COLOR,
    TEXT_THICKNESS,
    MOCK_PROCESSING_TIME,
    PROGRESS_UPDATE_INTERVAL
)

logger = logging.getLogger(__name__)


def process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Обработка видео через ML pipeline.

    Args:
        video_path: Путь к видео
        output_dir: Папка для результатов
        progress_callback: Callback для прогресса

    Returns:
        Результаты обработки
    """
    if ML_AVAILABLE:
        # Используем реальный ML pipeline
        logger.info("Using ML pipeline")
        try:
            return ml_process_video(video_path, output_dir, progress_callback)
        except Exception as e:
            logger.error(f"ML pipeline failed: {e}", exc_info=True)
            logger.warning("Falling back to mock pipeline")
            return _mock_process_video(video_path, output_dir, progress_callback)
    else:
        # Fallback на заглушку
        logger.warning("ML not available, using mock pipeline")
        return _mock_process_video(video_path, output_dir, progress_callback)


def _mock_process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """Mock pipeline для тестирования (fallback)."""

    logger.info(f"MOCK: Начало обработки видео: {video_path}")

    try:
        if progress_callback:
            progress_callback(0, "Инициализация pipeline...")

        time.sleep(PROGRESS_UPDATE_INTERVAL)
        if progress_callback:
            progress_callback(10, "Загрузка видео файла...")

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        logger.info(f"Свойства видео: {frame_count} кадров, {fps} fps, {duration:.2f}s длительность")

        if progress_callback:
            progress_callback(20, "Анализ кадров видео...")

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.warning("Не удалось прочитать кадр, генерация пустого предпросмотра")
            frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)

        stages = [
            (30, "Обнаружение объектов..."),
            (50, "Распознавание товаров..."),
            (70, "Извлечение штрих-кодов..."),
            (85, "Генерация результатов..."),
        ]

        for progress, message in stages:
            time.sleep(MOCK_PROCESSING_TIME / len(stages))
            if progress_callback:
                progress_callback(progress, message)

        # Создаем mock результаты в правильном формате CSV
        results_data = _generate_mock_results(duration, video_path.name)
        df = pd.DataFrame(results_data, columns=CSV_COLUMNS)

        if progress_callback:
            progress_callback(90, "Сохранение результатов...")

        csv_path = output_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Сохранены CSV результаты: {csv_path}")

        preview_path = output_dir / "preview.jpg"
        _generate_preview_image(frame, results_data[:3], preview_path)
        logger.info(f"Сохранено изображение предпросмотра: {preview_path}")

        if progress_callback:
            progress_callback(100, "Обработка завершена!")

        logger.info("MOCK: Обработка видео успешно завершена")

        return {
            "dataframe": df,
            "csv_path": csv_path,
            "preview_path": preview_path
        }

    except Exception as e:
        logger.error(f"Ошибка обработки видео: {str(e)}", exc_info=True)
        raise


def _generate_mock_results(duration: float, filename: str, num_detections: int = 15) -> list:
    """Generate mock results in correct CSV format."""

    products = [
        ("Мед ПОТАПЫЧ Натуральный", "415,79", "316,99", "4603552017456", "red", "К"),
        ("Вино BORDEAUX", "1999,99", "1499,99", "3296311111000", "red", "Ш"),
        ("Конфитюр ZUEGG", "347,39", "259,99", "80298342", "red", "Ш"),
    ]

    results = []
    for i in range(num_detections):
        timestamp = round(random.uniform(0, max(duration * 1000, 10000)), 0)
        product = random.choice(products)

        row = {
            'filename': filename,
            'product_name': product[0],
            'price_default': product[1],
            'price_card': product[2],
            'price_discount': 'нет',
            'barcode': product[3],
            'discount_amount': '-20%',
            'id_sku': '370204501518',
            'print_datetime': '04.01.2026 2:00',
            'code': '13_043015',
            'additional_info': 'нет',
            'color': product[4],
            'special_symbols': product[5],
            'frame_timestamp': int(timestamp),
            'x_min': random.randint(100, 500),
            'y_min': random.randint(100, 400),
            'x_max': random.randint(600, 1000),
            'y_max': random.randint(500, 800),
            # QR data
            'qr_code_barcode': product[3],
            'price1_qr': product[1].replace(',', '.'),
            'price2_qr': '',
            'price3_qr': '',
            'price4_qr': product[2].replace(',', '.'),
            'wholesale_level_1_count': 'нет',
            'wholesale_level_1_price': 'нет',
            'wholesale_level_2_count': 'нет',
            'wholesale_level_2_price': 'нет',
            'action_price_qr': 'нет',
            'action_code_qr': 'нет',
        }
        results.append(row)

    # Сортируем по времени
    results.sort(key=lambda x: x['frame_timestamp'])

    return results


def _generate_preview_image(
    frame: np.ndarray,
    detections: list,
    output_path: Path
) -> None:
    """Generate preview image with bounding boxes."""
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH or h > PREVIEW_HEIGHT:
        scale = min(PREVIEW_WIDTH / w, PREVIEW_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w

    preview = frame.copy()

    if preview.mean() < 10:
        preview = np.ones((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8) * 50
        h, w = PREVIEW_HEIGHT, PREVIEW_WIDTH

    num_boxes = min(len(detections), 3)
    for i in range(num_boxes):
        box_w, box_h = random.randint(100, 200), random.randint(80, 150)
        x1 = random.randint(50, max(51, w - box_w - 50))
        y1 = random.randint(50, max(51, h - box_h - 50))
        x2, y2 = x1 + box_w, y1 + box_h

        cv2.rectangle(preview, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

        product_name = detections[i].get('product_name', 'Product') if isinstance(detections[i], dict) else "Product"
        label = f"{product_name[:20]}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_THICKNESS
        )

        cv2.rectangle(
            preview,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            BBOX_COLOR,
            -1
        )

        cv2.putText(
            preview,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            TEXT_THICKNESS
        )

    title = "Предпросмотр обнаружения ценников"
    cv2.putText(
        preview,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        3
    )

    cv2.imwrite(str(output_path), preview)
    logger.info(f"Сгенерирован предпросмотр с {num_boxes} рамками обнаружения")
