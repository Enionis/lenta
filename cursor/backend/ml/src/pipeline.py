"""
Основной пайплайн обработки видео для распознавания ценников.

Интеграция с Backend через функцию process_video().
Архитектура: Детекция → Трекинг → Выбор лучших кадров → OCR → CSV
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple
import time
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .detection.detector import load_detector, draw_detections
from .utils.video_utils import get_video_info
from .utils.frame_quality import estimate_frame_quality
from .utils.color_detector import ColorDetector

# OCR (опционально)
try:
    from .ocr.ocr_processor import load_ocr_processor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR not available")

# QR (опционально)
try:
    from .qr.qr_decoder import load_qr_decoder
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    logger.warning("QR decoder not available")

logger = logging.getLogger(__name__)

# CSV колонки согласно заданию (30 полей)
CSV_COLUMNS = [
    # Данные с ценника
    'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
    'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
    'additional_info', 'color', 'special_symbols',
    'frame_timestamp', 'x_min', 'y_min', 'x_max', 'y_max',
    # Данные из QR
    'qr_code_barcode', 'price1_qr', 'price2_qr', 'price3_qr', 'price4_qr',
    'wholesale_level_1_count', 'wholesale_level_1_price',
    'wholesale_level_2_count', 'wholesale_level_2_price',
    'action_price_qr', 'action_code_qr'
]


def _build_barcode_rois(roi: np.ndarray) -> List[np.ndarray]:
    """
    Генерация ROI-кандидатов для штрихкода.
    Обычно штрихкод расположен в нижней части ценника, поэтому
    декодируем не только весь ROI, но и несколько полос.
    """
    if roi is None or roi.size == 0:
        return []

    h, w = roi.shape[:2]
    if h < 10 or w < 10:
        return [roi]

    candidates = [roi]
    y_slices = [
        (0.50, 1.00),
        (0.60, 1.00),
        (0.65, 0.98),
        (0.40, 0.90),
    ]

    for y0_ratio, y1_ratio in y_slices:
        y0 = max(0, int(h * y0_ratio))
        y1 = min(h, int(h * y1_ratio))
        if y1 - y0 < 8:
            continue
        crop = roi[y0:y1, :]
        if crop.size > 0:
            candidates.append(crop)

    # Центральная нижняя часть (иногда края обрезаны/зашумлены).
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    y0 = int(h * 0.55)
    y1 = int(h * 1.00)
    if x1 > x0 and y1 > y0:
        center_bottom = roi[y0:y1, x0:x1]
        if center_bottom.size > 0:
            candidates.append(center_bottom)

    return candidates


def _is_valid_barcode_candidate(value: str) -> bool:
    """Быстрая проверка кандидатного barcode."""
    if not value:
        return False
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return len(digits) == 13 and digits.startswith('46')


def _build_qr_rois(frame: Optional[np.ndarray], bbox: List[int], roi: Optional[np.ndarray]) -> List[np.ndarray]:
    """
    ROI-кандидаты для QR:
    - сам ценник,
    - расширенный bbox (контекст вокруг),
    - более крупный контекст вокруг ценника.
    """
    rois: List[np.ndarray] = []
    if roi is not None and roi.size > 0:
        rois.append(roi)

    if frame is None or frame.size == 0:
        return rois

    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)

    def crop_with_margin(margin_ratio: float):
        bw = max(1, x_max - x_min)
        bh = max(1, y_max - y_min)
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)
        x1 = max(0, x_min - mx)
        y1 = max(0, y_min - my)
        x2 = min(w, x_max + mx)
        y2 = min(h, y_max + my)
        if x2 > x1 and y2 > y1:
            c = frame[y1:y2, x1:x2]
            if c.size > 0:
                rois.append(c)

    # Контекст умеренный и широкий.
    crop_with_margin(0.20)
    crop_with_margin(0.45)
    return rois


class PriceTagTracker:
    """Трекер ценников с выбором лучших кадров."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.tracks = {}  # track_id -> {frames: [], best_detection: None}
        self.next_track_id = 1

    def iou(self, box1: List[float], box2: List[float]) -> float:
        """Вычисление IoU для двух bbox."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections: List[Dict], frame: np.ndarray, timestamp_ms: int):
        """Обновление треков новыми детекциями."""

        # Отмечаем все треки как необновленные
        updated_tracks = set()

        for det in detections:
            bbox = det['bbox']
            best_match = None
            best_iou = self.iou_threshold

            # Ищем лучшее совпадение по IoU
            for track_id, track_data in self.tracks.items():
                if track_id in updated_tracks:
                    continue

                last_bbox = track_data['last_bbox']
                iou_score = self.iou(bbox, last_bbox)

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match = track_id

            if best_match is not None:
                # Обновляем существующий трек
                track = self.tracks[best_match]
                track['last_bbox'] = bbox
                track['frames'].append({
                    'timestamp': timestamp_ms,
                    'bbox': bbox,
                    'conf': det.get('conf', 0),
                    'frame': frame.copy()  # Сохраняем кадр
                })
                updated_tracks.add(best_match)
            else:
                # Создаем новый трек
                self.tracks[self.next_track_id] = {
                    'last_bbox': bbox,
                    'frames': [{
                        'timestamp': timestamp_ms,
                        'bbox': bbox,
                        'conf': det.get('conf', 0),
                        'frame': frame.copy()
                    }]
                }
                updated_tracks.add(self.next_track_id)
                self.next_track_id += 1

    def get_best_detections(self) -> List[Dict]:
        """Получение лучших детекций для каждого трека."""

        best_detections = []

        for track_id, track_data in self.tracks.items():
            frames = track_data['frames']

            if not frames:
                continue
            # Выбираем кадр с лучшим качеством
            best_frame_idx = 0
            best_quality = 0

            for i, frame_data in enumerate(frames):
                frame = frame_data['frame']
                bbox = frame_data['bbox']

                # Оценка качества
                quality = estimate_frame_quality(frame, bbox)
                quality *= frame_data['conf']  # Учитываем уверенность детектора

                if quality > best_quality:
                    best_quality = quality
                    best_frame_idx = i

            best_frame_data = frames[best_frame_idx]

            best_detections.append({
                'track_id': track_id,
                'timestamp_ms': best_frame_data['timestamp'],
                'bbox': best_frame_data['bbox'],
                'conf': best_frame_data['conf'],
                'quality_score': best_quality,
                'num_frames': len(frames),
                'frame': best_frame_data['frame']
            })

        # Сортируем по времени
        best_detections.sort(key=lambda x: x['timestamp_ms'])

        return best_detections


def process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Основная функция обработки видео.

    Pipeline:
    1. Детекция ценников на каждом N-ом кадре
    2. Трекинг (сопоставление между кадрами)
    3. Выбор лучшего кадра для каждого ценника
    4. OCR + QR распознавание
    5. Сохранение в CSV

    Args:
        video_path: Путь к видео
        output_dir: Папка для результатов
        progress_callback: Callback прогресса

    Returns:
        Результаты обработки
    """
    start_time = time.time()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Начало обработки: {video_path}")

    def update_progress(progress: int, message: str):
        logger.info(f"[{progress}%] {message}")
        if progress_callback:
            progress_callback(progress, message)

    try:
        # ==== ЭТАП 1: Инициализация ====
        update_progress(0, "Инициализация...")

        detector = load_detector(use_gpu=True)
        tracker = PriceTagTracker()

        video_info = get_video_info(video_path)
        total_frames = video_info['frame_count']
        fps = video_info['fps']

        logger.info(f"Видео: {video_info['width']}x{video_info['height']}, "
                   f"{fps:.1f} fps, {total_frames} кадров")

        update_progress(5, "Анализ видео...")

        # ==== ЭТАП 2: Детекция и трекинг ====
        update_progress(10, "Детекция ценников...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_num = 0
        # Более плотный проход уменьшает пропуски детекции.
        # Приоритет этой задачи - качество, а не скорость.
        step = 3
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обрабатываем каждый N-й кадр
            if frame_num % step == 0:
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                # Детекция
                detections = detector.detect(frame, track=False)

                if detections:
                    # Обновление трекера
                    tracker.update(detections, frame, timestamp_ms)

                processed_frames += 1

                # Обновление прогресса
                if progress_callback and frame_num % (30 * step) == 0:
                    progress_pct = min(45, int(10 + (frame_num / total_frames) * 35))
                    update_progress(progress_pct,
                        f"Обработка кадров ({frame_num}/{total_frames})...")

            frame_num += 1

        cap.release()

        logger.info(f"Обработано {processed_frames} кадров, "
                   f"найдено {len(tracker.tracks)} треков")

        if not tracker.tracks:
            logger.warning("Ценники не обнаружены!")
            return create_empty_result(output_dir, video_path.name)

        # ==== ЭТАП 3: Выбор лучших кадров ====
        update_progress(50, "Выбор лучших кадров...")

        best_detections = tracker.get_best_detections()

        logger.info(f"Выбрано {len(best_detections)} лучших кадров для OCR")

        # ==== ЭТАП 4: OCR и парсинг ====
        update_progress(60, "Распознавание текста (OCR)...")

        # TODO: Добавить OCR в дни 8-10
        results = process_detections_ocr(best_detections, video_path.name)

        update_progress(80, "Генерация результатов...")

        # ==== ЭТАП 5: Сохранение ====
        df = pd.DataFrame(results, columns=CSV_COLUMNS)

        csv_path = output_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"CSV сохранен: {csv_path} ({len(df)} записей)")

        # ==== ЭТАП 6: Превью ====
        update_progress(90, "Создание превью...")

        preview_path = generate_preview(
            video_path=video_path,
            best_detections=best_detections[:5],
            output_path=output_dir / "preview.jpg"
        )

        elapsed = time.time() - start_time
        logger.info(f"Обработка завершена за {elapsed:.1f} сек")

        update_progress(100, "Готово!")

        return {
            'dataframe': df,
            'csv_path': csv_path,
            'preview_path': preview_path,
            'stats': {
                'processing_time': elapsed,
                'frames_processed': processed_frames,
                'tracks_found': len(tracker.tracks),
                'results_count': len(df)
            }
        }

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
        raise


def process_detections_ocr(detections: List[Dict], filename: str) -> List[Dict]:
    """
    OCR обработка детекций.
    """
    results = []

    # Инициализация OCR / QR / Color
    ocr_processor = None
    qr_decoder = None
    color_detector = ColorDetector()

    if OCR_AVAILABLE:
        try:
            ocr_processor = load_ocr_processor(use_gpu=True)
            logger.info("OCR processor loaded")
        except Exception as e:
            logger.warning(f"Failed to load OCR: {e}")

    if QR_AVAILABLE:
        try:
            qr_decoder = load_qr_decoder()
            logger.info("QR decoder loaded")
        except Exception as e:
            logger.warning(f"Failed to load QR decoder: {e}")

    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det['bbox'])

        # Получаем ROI (Region of Interest)
        frame = det.get('frame')
        roi = None
        if frame is not None:
            h, w = frame.shape[:2]
            # Проверяем границы
            x_min = max(0, min(x_min, w))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h))
            y_max = max(0, min(y_max, h))

            if x_max > x_min and y_max > y_min:
                roi = frame[y_min:y_max, x_min:x_max]

        # OCR обработка
        ocr_result = {
            'product_name': '',
            'price_default': '',
            'price_card': '',
            'barcode': '',
            'id_sku': '',
            'discount_amount': '',
            'print_datetime': '',
            'code': '',
            'special_symbols': '',
            'all_texts': []
        }

        qr_result = {
            'qr_code_barcode': '',
            'price1_qr': '',
            'price2_qr': '',
            'price3_qr': '',
            'price4_qr': '',
            'wholesale_level_1_count': '',
            'wholesale_level_1_price': '',
            'wholesale_level_2_count': '',
            'wholesale_level_2_price': '',
            'action_price_qr': '',
            'action_code_qr': ''
        }

        if ocr_processor is not None and roi is not None:
            try:
                ocr_data = ocr_processor.extract_price_tag_info(roi)
                ocr_result = {
                    'product_name': ocr_data.get('product_name', ''),
                    'price_default': ocr_data.get('price_default', ''),
                    'price_card': ocr_data.get('price_card', ''),
                    'barcode': ocr_data.get('barcode', ''),
                    'id_sku': ocr_data.get('id_sku', ''),
                    'discount_amount': ocr_data.get('discount_amount', ''),
                    'print_datetime': ocr_data.get('print_datetime', ''),
                    'code': ocr_data.get('code', ''),
                    'special_symbols': ocr_data.get('special_symbols', ''),
                    'all_texts': ocr_data.get('all_texts', [])
                }
            except Exception as e:
                logger.warning(f"OCR failed for track {det['track_id']}: {e}")

        if qr_decoder is not None and roi is not None:
            try:
                # 1) QR-pass по нескольким ROI (сам ценник + контекст вокруг)
                qr_rois = _build_qr_rois(frame, det['bbox'], roi)
                for qr_roi in qr_rois:
                    qr_decoded = qr_decoder.decode(qr_roi)
                    for item in qr_decoded:
                        qr_parsed = qr_decoder.parse_lenta_qr(item.get('data', ''))
                        for k, v in qr_parsed.items():
                            if not qr_result.get(k) and v:
                                qr_result[k] = v

                # 2) Отдельный barcode-pass по локальным ROI.
                if not qr_result.get('qr_code_barcode'):
                    for barcode_roi in _build_barcode_rois(roi):
                        barcode_candidate = qr_decoder.extract_barcode(barcode_roi)
                        if _is_valid_barcode_candidate(barcode_candidate):
                            qr_result['qr_code_barcode'] = barcode_candidate
                            break

                # 3) Специализированный 1D-pass для barcode.
                if not qr_result.get('qr_code_barcode'):
                    for barcode_roi in _build_barcode_rois(roi):
                        barcode_candidate = qr_decoder.extract_barcode_1d(barcode_roi)
                        if _is_valid_barcode_candidate(barcode_candidate):
                            qr_result['qr_code_barcode'] = barcode_candidate
                            break
            except Exception as e:
                logger.warning(f"QR failed for track {det['track_id']}: {e}")

        # Цвет определяем всегда, если есть ROI
        tag_color = 'unknown'
        if roi is not None:
            try:
                tag_color = color_detector.detect(roi)
            except Exception:
                tag_color = 'unknown'

        # Для primary key приоритет QR barcode (обычно более надежен).
        ocr_barcode = ocr_result.get('barcode', '')
        qr_barcode = qr_result.get('qr_code_barcode', '')
        if _is_valid_barcode_candidate(qr_barcode):
            barcode_value = qr_barcode
        elif _is_valid_barcode_candidate(ocr_barcode):
            barcode_value = ocr_barcode
        else:
            # Если EAN-13 невалиден, не используем его как ключ.
            barcode_value = ''

        # Вычисляем price_discount
        price_discount = 'нет'
        try:
            if ocr_result['price_default'] and ocr_result['price_card']:
                p_default = float(str(ocr_result['price_default']).replace(',', '.'))
                p_card = float(str(ocr_result['price_card']).replace(',', '.'))
                if p_default > p_card > 0:
                    discount_pct = round((1 - p_card / p_default) * 100)
                    price_discount = f"-{discount_pct}%"
        except Exception:
            price_discount = 'нет'

        additional_info = ' '.join(ocr_result.get('all_texts', []))[:500] if ocr_result.get('all_texts') else 'нет'

        row = {
            'filename': filename,
            'product_name': ocr_result['product_name'],
            'price_default': ocr_result['price_default'],
            'price_card': ocr_result['price_card'],
            'price_discount': price_discount,
            'barcode': barcode_value,
            'discount_amount': ocr_result['discount_amount'],
            'id_sku': ocr_result['id_sku'],
            'print_datetime': ocr_result['print_datetime'],
            'code': ocr_result['code'],
            'additional_info': additional_info,
            'color': tag_color,
            'special_symbols': ocr_result['special_symbols'],
            'frame_timestamp': int(det['timestamp_ms']),
            'x_min': float(x_min),
            'y_min': float(y_min),
            'x_max': float(x_max),
            'y_max': float(y_max),
            'qr_code_barcode': qr_result['qr_code_barcode'],
            'price1_qr': qr_result['price1_qr'],
            'price2_qr': qr_result['price2_qr'],
            'price3_qr': qr_result['price3_qr'],
            'price4_qr': qr_result['price4_qr'],
            'wholesale_level_1_count': qr_result['wholesale_level_1_count'],
            'wholesale_level_1_price': qr_result['wholesale_level_1_price'],
            'wholesale_level_2_count': qr_result['wholesale_level_2_count'],
            'wholesale_level_2_price': qr_result['wholesale_level_2_price'],
            'action_price_qr': qr_result['action_price_qr'],
            'action_code_qr': qr_result['action_code_qr'],
        }
        results.append(row)

    return results


def generate_preview(
    video_path: Path,
    best_detections: List[Dict],
    output_path: Path
) -> Path:
    """Создание превью с детекциями."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        preview = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        cv2.putText(preview, "No preview available", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(output_path), preview)
        return output_path

    # Берем первый кадр с детекциями
    if best_detections:
        det = best_detections[0]
        cap.set(cv2.CAP_PROP_POS_MSEC, det['timestamp_ms'])
        ret, frame = cap.read()

        if ret and frame is not None:
            # Отрисовка всех bbox на этом кадре
            for d in best_detections:
                # Проверяем что кадр тот же (±1000 мс)
                if abs(d['timestamp_ms'] - det['timestamp_ms']) < 1000:
                    x1, y1, x2, y2 = map(int, d['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{d['track_id']}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Масштабирование
            max_size = 1280
            h, w = frame.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imwrite(str(output_path), frame)
            cap.release()
            return output_path

    cap.release()

    # Fallback
    preview = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    cv2.imwrite(str(output_path), preview)
    return output_path


def create_empty_result(output_dir: Path, filename: str) -> Dict[str, Any]:
    """Создание пустого результата."""
    df = pd.DataFrame(columns=CSV_COLUMNS)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    preview_path = output_dir / "preview.jpg"
    preview = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    cv2.putText(preview, "No price tags detected", (400, 360),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(str(preview_path), preview)

    return {
        'dataframe': df,
        'csv_path': csv_path,
        'preview_path': preview_path,
        'stats': {'tracks_found': 0, 'results_count': 0}
    }
