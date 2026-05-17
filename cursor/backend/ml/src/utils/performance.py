"""Performance optimization utilities."""

import time
import logging
from functools import wraps
from typing import Callable, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class Timer:
    """Контекстный менеджер для измерения времени."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name}: {self.elapsed:.3f}s")


def profile_function(func: Callable) -> Callable:
    """Декоратор для профилирования функций."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


class FrameSkipper:
    """Умный пропуск кадров для ускорения обработки."""

    def __init__(self, target_fps: float = 2.0, video_fps: float = 30.0):
        """
        Args:
            target_fps: Целевой FPS обработки
            video_fps: FPS исходного видео
        """
        self.step = max(1, int(video_fps / target_fps))
        self.frame_count = 0

    def should_process(self, frame_num: int) -> bool:
        """Проверить, нужно ли обрабатывать этот кадр."""
        return frame_num % self.step == 0


class BatchProcessor:
    """Батчевая обработка для ускорения."""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.buffer = []

    def add(self, item: Any) -> bool:
        """Добавить элемент в буфер."""
        self.buffer.append(item)
        return len(self.buffer) >= self.batch_size

    def get_batch(self) -> list:
        """Получить текущий батч и очистить буфер."""
        batch = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        return batch

    def flush(self) -> list:
        """Получить оставшиеся элементы."""
        remaining = self.buffer
        self.buffer = []
        return remaining


def optimize_for_inference(model_path: str, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Оптимизация модели для инференса.

    Returns:
        Конфигурация для оптимизированного инференса
    """
    config = {
        'half_precision': use_gpu,  # FP16 для GPU
        'batch_size': 1,
        'imgsz': 640,
        'conf_thres': 0.5,
        'iou_thres': 0.45,
        'max_det': 100,
    }

    # Проверяем доступность GPU
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                config['device'] = 'cuda'
            else:
                config['device'] = 'cpu'
                config['half_precision'] = False
        except ImportError:
            config['device'] = 'cpu'
            config['half_precision'] = False

    return config


def estimate_processing_time(
    video_duration_sec: float,
    processing_fps: float = 2.0
) -> Dict[str, float]:
    """Оценка времени обработки видео."""

    total_frames = video_duration_sec * processing_fps

    # Временные оценки для каждого этапа
    times = {
        'detection': total_frames * 0.05,  # ~50ms на кадр
        'tracking': total_frames * 0.01,   # ~10ms
        'ocr_per_tag': 0.5,                 # ~500ms на ценник
        'qr_per_tag': 0.1,                  # ~100ms на ценник
        'color_per_tag': 0.01,              # ~10ms
    }

    # Предполагаем ~50 ценников на минуту видео
    estimated_tags = (video_duration_sec / 60) * 50

    total_time = (
        times['detection'] +
        times['tracking'] +
        estimated_tags * (times['ocr_per_tag'] + times['qr_per_tag'] + times['color_per_tag'])
    )

    return {
        'total_seconds': total_time,
        'detection_seconds': times['detection'],
        'postprocessing_seconds': estimated_tags * (times['ocr_per_tag'] + times['qr_per_tag']),
        'estimated_tags': int(estimated_tags)
    }
