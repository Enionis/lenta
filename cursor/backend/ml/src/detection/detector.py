"""Price tag detector using YOLO."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Попытка импорта YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available. Detector will not work.")


class PriceTagDetector:
    """Детектор ценников на базе YOLO."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Инициализация детектора.

        Args:
            model_path: Путь к модели YOLO (.pt или .onnx)
            conf_threshold: Порог уверенности
            iou_threshold: Порог IoU для NMS
            device: 'cuda', 'cpu', или 'auto'
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not installed. Run: pip install ultralytics")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Загрузка модели
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
        else:
            # Ищем обученную модель
            trained_model = Path('backend/ml/models/price_tag_detector.pt')
            if trained_model.exists():
                logger.info(f"Loading trained price tag detector: {trained_model}")
                self.model = YOLO(str(trained_model))
            else:
                # Fallback на предобученную YOLOv8n
                logger.info("Loading pretrained YOLOv8n (COCO)")
                logger.warning("For price tag detection, you should train or download a custom model!")
                self.model = YOLO('yolov8n.pt')

    def detect(
        self,
        frame: np.ndarray,
        track: bool = True
    ) -> List[Dict]:
        """
        Детекция ценников на кадре.

        Args:
            frame: BGR изображение (numpy array)
            track: Использовать трекинг (ByteTrack)

        Returns:
            Список словарей с детекциями:
            [{
                'bbox': [x_min, y_min, x_max, y_max],
                'conf': 0.95,
                'track_id': 1 (если track=True)
            }, ...]
        """
        if frame is None or frame.size == 0:
            return []

        # Запуск инференса
        if track:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

        if not results or len(results) == 0:
            return []

        detections = []
        result = results[0]  # Первый результат

        if result.boxes is None:
            return []

        boxes = result.boxes

        frame_h, frame_w = frame.shape[:2]

        for i, box in enumerate(boxes):
            # Координаты bbox
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

            # Уверенность
            conf = float(box.conf[0].cpu().numpy())

            # Track ID (если используется трекинг)
            track_id = None
            if track and box.id is not None:
                track_id = int(box.id[0].cpu().numpy())

            detection = {
                'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
                'conf': conf,
                'track_id': track_id
            }

            if not self._is_reasonable_bbox(detection['bbox'], conf, frame_w, frame_h):
                continue

            detections.append(detection)

        return detections

    def _is_reasonable_bbox(
        self,
        bbox: List[float],
        conf: float,
        frame_w: int,
        frame_h: int
    ) -> bool:
        """Фильтрация явных ложных срабатываний детектора."""
        if conf < max(0.30, self.conf_threshold - 0.05):
            return False

        x_min, y_min, x_max, y_max = bbox
        bw = max(1.0, x_max - x_min)
        bh = max(1.0, y_max - y_min)
        area = bw * bh
        frame_area = max(1.0, float(frame_w * frame_h))
        area_ratio = area / frame_area
        aspect = bw / bh

        # Типичный ценник занимает небольшую часть кадра и не бывает
        # слишком узким или чрезмерно "плашкой" по форме.
        if area_ratio < 0.0005 or area_ratio > 0.16:
            return False
        if bw < 65 or bh < 38:
            return False
        if aspect < 0.35 or aspect > 4.8:
            return False

        return True

    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """Батчевая детекция для ускорения."""
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model(batch, conf=self.conf_threshold, verbose=False)

            for result in results:
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append({
                            'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
                            'conf': conf,
                            'track_id': None
                        })
                all_detections.append(detections)

        return all_detections


def load_detector(
    model_path: Optional[Path] = None,
    use_gpu: bool = True
) -> PriceTagDetector:
    """
    Фабричная функция для загрузки детектора.

    Args:
        model_path: Путь к модели
        use_gpu: Использовать GPU если доступен

    Returns:
        Инициализированный PriceTagDetector
    """
    device = 'cuda' if use_gpu else 'cpu'

    # Поиск модели по умолчанию
    if model_path is None:
        possible_paths = [
            Path('backend/ml/models/price_tag_detector.pt'),
            Path('backend/ml/models/price_tag_detector.onnx'),
            Path('models/price_tag_detector.pt'),
        ]

        for path in possible_paths:
            if path.exists():
                model_path = path
                logger.info(f"Found model at {path}")
                break

    # Пробуем ONNX если файл .onnx и не требуется GPU
    if model_path and model_path.suffix == '.onnx' and not use_gpu:
        try:
            from .onnx_detector import load_onnx_detector
            return load_onnx_detector(str(model_path))
        except Exception as e:
            logger.warning(f"ONNX loading failed: {e}")

    return PriceTagDetector(model_path=model_path, device=device)


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
    show_id: bool = True
) -> np.ndarray:
    """Отрисовка детекций на кадре."""
    output = frame.copy()

    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det['bbox'])
        conf = det.get('conf', 0)
        track_id = det.get('track_id')

        # Рамка
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, thickness)

        # Подпись
        labels = []
        if show_id and track_id is not None:
            labels.append(f"ID:{track_id}")
        if show_conf:
            labels.append(f"{conf:.2f}")

        if labels:
            label = " ".join(labels)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Фон для текста
            cv2.rectangle(
                output,
                (x_min, y_min - text_h - 8),
                (x_min + text_w + 8, y_min),
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (x_min + 4, y_min - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    return output
