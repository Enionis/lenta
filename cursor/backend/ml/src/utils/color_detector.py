"""Color detection for price tags using HSV color space."""

import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ColorDetector:
    """Детектор цвета ценника (red, white, yellow, blue)."""

    # HSV диапазоны для разных цветов
    COLOR_RANGES = {
        'red': [
            # Красный проходит через 0/180, поэтому два диапазона
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        'white': [
            (np.array([0, 0, 200]), np.array([180, 30, 255]))
        ],
        'yellow': [
            (np.array([20, 100, 100]), np.array([40, 255, 255]))
        ],
        'blue': [
            (np.array([100, 100, 100]), np.array([140, 255, 255]))
        ],
        'green': [
            (np.array([40, 100, 100]), np.array([80, 255, 255]))
        ]
    }

    def __init__(self, min_area_ratio: float = 0.1):
        """
        Args:
            min_area_ratio: Минимальная доля пикселей цвета для определения
        """
        self.min_area_ratio = min_area_ratio

    def detect(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> str:
        """
        Определение цвета ценника.

        Args:
            image: Изображение (BGR)
            bbox: Опционально - ограничивающая рамка (x_min, y_min, x_max, y_max)

        Returns:
            Цвет: 'red', 'white', 'yellow', 'blue', 'unknown'
        """
        if image is None or image.size == 0:
            return 'unknown'

        # Вырезаем ROI если задан bbox
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            h, w = image.shape[:2]

            # Проверяем границы
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(w, int(x_max))
            y_max = min(h, int(y_max))

            if x_max <= x_min or y_max <= y_min:
                return 'unknown'

            roi = image[y_min:y_max, x_min:x_max]
        else:
            roi = image

        # Конвертация в HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Подсчет пикселей каждого цвета
        color_scores = {}
        total_pixels = roi.shape[0] * roi.shape[1]

        for color_name, ranges in self.COLOR_RANGES.items():
            mask = None

            for lower, upper in ranges:
                range_mask = cv2.inRange(hsv, lower, upper)

                if mask is None:
                    mask = range_mask
                else:
                    mask = cv2.bitwise_or(mask, range_mask)

            # Подсчет пикселей
            pixel_count = cv2.countNonZero(mask)
            ratio = pixel_count / total_pixels

            color_scores[color_name] = ratio

        # Определяем доминирующий цвет
        if not color_scores:
            return 'unknown'

        best_color = max(color_scores, key=color_scores.get)
        best_score = color_scores[best_color]

        # Проверяем порог
        if best_score < self.min_area_ratio:
            # Возможно белый фон
            if color_scores.get('white', 0) > 0.3:
                return 'white'
            return 'unknown'

        return best_color

    def detect_with_confidence(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> Dict:
        """
        Определение цвета с уверенностью для всех классов.

        Returns:
            {
                'color': лучший цвет,
                'confidence': уверенность (0-1),
                'scores': {'red': 0.5, 'white': 0.1, ...}
            }
        """
        if image is None or image.size == 0:
            return {'color': 'unknown', 'confidence': 0, 'scores': {}}

        # Вырезаем ROI
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            h, w = image.shape[:2]
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

            if x_max > x_min and y_max > y_min:
                roi = image[y_min:y_max, x_min:x_max]
            else:
                roi = image
        else:
            roi = image

        # HSV анализ
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_scores = {}
        total_pixels = roi.shape[0] * roi.shape[1]

        for color_name, ranges in self.COLOR_RANGES.items():
            mask = None
            for lower, upper in ranges:
                range_mask = cv2.inRange(hsv, lower, upper)
                if mask is None:
                    mask = range_mask
                else:
                    mask = cv2.bitwise_or(mask, range_mask)

            pixel_count = cv2.countNonZero(mask)
            color_scores[color_name] = pixel_count / total_pixels

        # Нормализация в вероятности
        total_score = sum(color_scores.values())
        if total_score > 0:
            probs = {k: v/total_score for k, v in color_scores.items()}
        else:
            probs = {k: 0 for k in color_scores}
            probs['unknown'] = 1.0

        best_color = max(probs, key=probs.get)

        return {
            'color': best_color,
            'confidence': probs[best_color],
            'scores': probs
        }


def detect_price_tag_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """Удобная функция для определения цвета ценника."""
    detector = ColorDetector()
    return detector.detect(image, bbox)
