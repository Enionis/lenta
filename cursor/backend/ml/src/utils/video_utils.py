"""Video processing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def get_video_info(video_path: Path) -> Dict:
    """Получение информации о видео файле."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }

    cap.release()
    return info


def extract_frames(
    video_path: Path,
    output_dir: Path,
    step: int = 1,
    max_frames: Optional[int] = None
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Извлечение кадров из видео.

    Args:
        video_path: Путь к видео
        output_dir: Папка для сохранения (опционально)
        step: Шаг извлечения (каждый N-й кадр)
        max_frames: Максимальное количество кадров

    Yields:
        (frame_number, frame) - номер кадра и изображение
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_num = 0
    saved_count = 0

    logger.info(f"Извлечение кадров из {video_path} (step={step})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % step == 0:
            yield frame_num, frame
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_num += 1

    cap.release()
    logger.info(f"Извлечено {saved_count} кадров")


def save_frame(frame: np.ndarray, output_path: Path, quality: int = 95) -> None:
    """Сохранение кадра в файл."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), frame)
