"""Frame quality estimation utilities."""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def estimate_frame_quality(frame: np.ndarray, bbox: List[float]) -> float:
    """
    Оценка качества кадра для OCR.

    Использует комбинацию:
    - Лапласиан (резкость)
    - Площадь региона
    - Контраст

    Args:
        frame: Изображение (BGR)
        bbox: [x_min, y_min, x_max, y_max] в пикселях

    Returns:
        score: чем выше, тем лучше качество
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Проверка границ
    h, w = frame.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    # Вырезаем ROI
    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return 0.0

    # Конвертация в grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Лапласиан (резкость)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Площадь (больше = лучше для OCR)
    area = (x_max - x_min) * (y_max - y_min)
    area_score = np.log(area + 1)

    # 3. Контраст (стандартное отклонение яркости)
    contrast = np.std(gray)

    # 4. Проверка на размытость движения (простая эвристика)
    # Если много высокочастотных компонент - кадр резкий
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq = np.sum(magnitude > np.percentile(magnitude, 90))

    # Итоговый score (взвешенная комбинация)
    # Нормализация значений
    laplacian_norm = min(laplacian_var / 500, 10)  # Нормируем ~0-10
    contrast_norm = contrast / 50  # Нормируем ~0-5

    score = laplacian_norm * 0.4 + area_score * 0.3 + contrast_norm * 0.2 + (high_freq / 1000) * 0.1

    return float(score)


def select_best_frames(
    frame_history: List[Tuple[int, np.ndarray, float]],
    top_k: int = 3
) -> List[Tuple[int, np.ndarray, float]]:
    """
    Выбор лучших кадров из истории.

    Args:
        frame_history: список (frame_num, frame, quality_score)
        top_k: сколько лучших выбрать

    Returns:
        лучшие кадры отсортированные по качеству
    """
    if not frame_history:
        return []

    # Сортировка по score (убывание)
    sorted_frames = sorted(frame_history, key=lambda x: x[2], reverse=True)

    return sorted_frames[:top_k]


def is_frame_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """Проверка на размытость кадра."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold
