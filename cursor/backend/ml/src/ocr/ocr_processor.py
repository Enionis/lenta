"""OCR processing using PaddleOCR."""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. OCR will not work.")

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR процессор для распознавания текста с ценников."""

    def __init__(
        self,
        lang: str = 'ru',
        use_gpu: bool = True,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        rec_char_dict_path: Optional[str] = None,
        enable_fallback_general_ocr: bool = True
    ):
        """
        Инициализация OCR.

        Args:
            lang: Язык (ru, en, etc.)
            use_gpu: Использовать GPU
            det_model_dir: Путь к модели детекции текста
            rec_model_dir: Путь к модели распознавания текста
            rec_char_dict_path: Путь к словарю символов для rec модели
            enable_fallback_general_ocr: Второй OCR-контур общей модели
        """
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not installed. Run: pip install paddleocr")

        self.lang = lang

        # Инициализация PaddleOCR
        kwargs = {
            'use_angle_cls': True,
            'lang': lang
        }

        if det_model_dir:
            kwargs['det_model_dir'] = det_model_dir
        if rec_model_dir:
            kwargs['rec_model_dir'] = rec_model_dir
        if rec_char_dict_path:
            kwargs['rec_char_dict_path'] = rec_char_dict_path

        logger.info(f"Initializing PaddleOCR (lang={lang}, use_gpu={use_gpu})")
        try:
            self.ocr = PaddleOCR(**kwargs)
        except TypeError as e:
            # Некоторые версии PaddleOCR не принимают часть аргументов.
            if 'use_gpu' in str(e):
                logger.warning("PaddleOCR doesn't accept use_gpu parameter, falling back.")
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    det_model_dir=det_model_dir or None,
                    rec_model_dir=rec_model_dir or None,
                    rec_char_dict_path=rec_char_dict_path or None
                )
            else:
                raise

        # Дополнительный OCR-контур:
        # если используем кастомный rec_model_dir, запускаем вторую (дефолтную) модель
        # для извлечения текстовых строк (название товара и т.п.).
        self.fallback_ocr = None
        if enable_fallback_general_ocr and rec_model_dir:
            try:
                self.fallback_ocr = PaddleOCR(use_angle_cls=True, lang=lang)
                logger.info("Fallback general OCR initialized")
            except Exception as e:
                logger.warning(f"Failed to init fallback OCR: {e}")

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Извлечение текста из изображения.

        Args:
            image: numpy array (BGR или RGB)

        Returns:
            Список словарей с текстом и координатами:
            [{
                'text': 'распознанный текст',
                'confidence': 0.95,
                'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            }, ...]
        """
        if image is None or image.size == 0:
            return []

        # Конвертация BGR → RGB если нужно
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image

        # Базовое OCR
        result = self.ocr.ocr(image_rgb, cls=True)
        lines = self._normalize_ocr_lines(result)

        # Fallback: если ничего не нашли, пробуем усиленную предобработку.
        if not lines:
            enhanced = self._enhance_for_ocr(image_rgb)
            result = self.ocr.ocr(enhanced, cls=True)
            lines = self._normalize_ocr_lines(result)

        # Fallback OCR-контур (дефолтная модель) для текстовых полей.
        if self.fallback_ocr is not None:
            try:
                fallback_result = self.fallback_ocr.ocr(image_rgb, cls=True)
                fallback_lines = self._normalize_ocr_lines(fallback_result)
                if fallback_lines:
                    lines = self._merge_lines(lines, fallback_lines)
            except Exception:
                pass

        if not lines:
            return []

        # Обработка результатов
        texts = []
        for line in lines:
            if line is None or len(line) < 2:
                continue

            bbox = line[0]
            text_info = line[1]
            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue

            text, conf = text_info[0], text_info[1]
            if text is None:
                continue
            conf_value = float(conf) if conf is not None else 0.0
            if conf_value < 0.40:
                continue

            texts.append({
                'text': str(text),
                'confidence': conf_value,
                'bbox': bbox
            })

        return texts

    def _merge_lines(self, primary_lines: List, secondary_lines: List) -> List:
        """
        Объединение OCR-линий из двух контуров, с дедупликацией по text+bbox.
        """
        merged = []
        seen = set()

        def _key(item):
            try:
                bbox = item[0]
                text = item[1][0]
                if isinstance(bbox, list) and bbox:
                    flat_bbox = tuple(int(v) for pt in bbox for v in pt[:2])
                else:
                    flat_bbox = ()
                return (str(text).strip().lower(), flat_bbox)
            except Exception:
                return None

        for source in (primary_lines or [], secondary_lines or []):
            for item in source:
                key = _key(item)
                if key is None or key in seen:
                    continue
                seen.add(key)
                merged.append(item)

        return merged

    def _enhance_for_ocr(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Усиление контраста и резкости для мелкого текста на ценниках.
        """
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb

        # Увеличение ROI (для мелкого шрифта)
        h, w = image_rgb.shape[:2]
        scale = 2.0 if min(h, w) < 120 else 1.5
        resized = cv2.resize(
            image_rgb,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC
        )

        # Локальное повышение контраста в L-канале
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Легкое шумоподавление
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        return enhanced

    def _normalize_ocr_lines(self, result) -> List:
        """
        Приведение результата PaddleOCR к списку OCR-линий.

        Разные версии PaddleOCR могут возвращать:
        - [ [line1, line2, ...] ]
        - [line1, line2, ...]
        - None / пустые структуры
        """
        if not result:
            return []

        if isinstance(result, list):
            lines = result

            # Распаковываем лишние уровни вложенности: [[lines]] -> [lines]
            # и т.д., пока структура не станет списком OCR-линий.
            while (
                isinstance(lines, list)
                and len(lines) == 1
                and isinstance(lines[0], list)
            ):
                nested = lines[0]
                if not nested:
                    return []
                # Если nested похож на список OCR-линий, разворачиваем.
                if isinstance(nested[0], (list, tuple)):
                    lines = nested
                else:
                    break

            if not isinstance(lines, list) or not lines:
                return []

            # Оставляем только элементы, похожие на OCR-линию:
            # [bbox, (text, conf)]
            normalized = []
            for item in lines:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                text_info = item[1]
                if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                    continue
                normalized.append(item)

            return normalized

        return []

    def extract_price_tag_info(self, image: np.ndarray) -> Dict:
        """
        Извлечение структурированной информации с ценника.

        Args:
            image: Изображение ценника

        Returns:
            Словарь с полями ценника
        """
        texts = self.extract_text(image)
        name_texts = self._extract_name_pass(image)
        texts = self._merge_text_items(texts, name_texts)

        # Объединяем все тексты для анализа
        all_text_lines = [t['text'] for t in texts]

        # Используем пост-обработку для извлечения полей
        try:
            from .text_postprocess import OCRPostProcessor
            processor = OCRPostProcessor()
            result = processor.extract_fields(all_text_lines, text_items=texts)
        except Exception as e:
            # Fallback на базовое извлечение
            result = {
                'product_name': self._extract_product_name(texts),
                'price_default': self._extract_price(texts, 'default'),
                'price_card': self._extract_price(texts, 'card'),
                'barcode': self._extract_barcode(texts),
                'id_sku': self._extract_sku(texts),
                'discount_amount': self._extract_discount(texts),
            }

        result['all_texts'] = all_text_lines
        return result

    def _extract_name_pass(self, image: np.ndarray) -> List[Dict]:
        """
        Дополнительный OCR-проход по верхней части ценника для product_name.
        """
        if image is None or image.size == 0:
            return []
        if self.fallback_ocr is None:
            return []

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []

        # Верхняя зона обычно содержит название товара.
        y0 = int(h * 0.00)
        y1 = int(h * 0.55)
        top_roi = image[y0:y1, :]
        if top_roi.size == 0:
            return []

        # RGB + мягкая предобработка только для name-pass.
        top_rgb = top_roi[:, :, ::-1] if len(top_roi.shape) == 3 else top_roi
        top_enhanced = self._enhance_name_region(top_rgb)

        lines = []
        try:
            raw = self.fallback_ocr.ocr(top_rgb, cls=True)
            lines.extend(self._normalize_ocr_lines(raw))
        except Exception:
            pass

        try:
            raw_enh = self.fallback_ocr.ocr(top_enhanced, cls=True)
            lines.extend(self._normalize_ocr_lines(raw_enh))
        except Exception:
            pass

        if not lines:
            return []

        text_items = []
        for line in lines:
            if line is None or len(line) < 2:
                continue
            bbox = line[0]
            text_info = line[1]
            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue
            text, conf = text_info[0], text_info[1]
            if text is None:
                continue
            text_items.append({
                'text': str(text),
                'confidence': float(conf) if conf is not None else 0.0,
                'bbox': bbox
            })

        return text_items

    def _enhance_name_region(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Мягкое улучшение верхней зоны ценника под распознавание названия.
        """
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb

        # Увеличиваем умеренно, чтобы не разрушать буквы.
        h, w = image_rgb.shape[:2]
        scale = 1.8 if min(h, w) < 120 else 1.4
        resized = cv2.resize(
            image_rgb,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC
        )

        # CLAHE на яркостном канале.
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        # Легкий denoise.
        out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)
        return out

    def _merge_text_items(self, base_items: List[Dict], extra_items: List[Dict]) -> List[Dict]:
        """
        Объединение OCR text_items с дедупликацией по тексту.
        """
        best_by_text: Dict[str, Dict] = {}

        for item in (base_items or []) + (extra_items or []):
            text = str(item.get('text', '')).strip()
            if not text:
                continue
            key = text.lower()
            conf = float(item.get('confidence', 0.0) or 0.0)
            prev = best_by_text.get(key)
            if prev is None or conf > float(prev.get('confidence', 0.0) or 0.0):
                best_by_text[key] = item

        return list(best_by_text.values())

    def _extract_product_name(self, texts: List[Dict]) -> str:
        """Извлечение наименования товара."""
        # Ищем длинную строку без цифр (название продукта)
        candidates = []

        for t in texts:
            text = t['text']
            # Фильтруем короткие строки и строки с ценами/штрихкодами
            if len(text) > 10 and not re.match(r'^[\d\s,.]+$', text):
                if not re.search(r'\d{13}', text):  # Не штрихкод
                    if not re.search(r'\d+[,.]\d{2}', text):  # Не цена
                        candidates.append(text)

        return candidates[0] if candidates else ''

    def _extract_price(self, texts: List[Dict], price_type: str = 'default') -> str:
        """Извлечение цены."""
        prices = []

        for t in texts:
            text = t['text'].replace(',', '.').replace(' ', '')

            # Ищем числа формата XXX.XX или XXXX.XX
            matches = re.findall(r'(\d+[.]\d{2})', text)
            prices.extend(matches)

        if not prices:
            return ''

        # Конвертируем в float
        price_values = [float(p) for p in prices]

        if price_type == 'default':
            # Большая цена (обычно без скидки)
            return str(max(price_values)) if price_values else ''
        elif price_type == 'card':
            # Меньшая цена (по карте)
            if len(price_values) >= 2:
                return str(min(price_values))
            return ''

        return ''

    def _extract_barcode(self, texts: List[Dict]) -> str:
        """Извлечение штрихкода (13 цифр)."""
        for t in texts:
            text = t['text'].replace(' ', '')

            # Ищем 13 цифр
            match = re.search(r'(\d{13})', text)
            if match:
                barcode = match.group(1)
                # Проверяем что начинается с 46 (Россия)
                if barcode.startswith('46'):
                    return barcode

        return ''

    def _extract_sku(self, texts: List[Dict]) -> str:
        """Извлечение артикула (12 цифр)."""
        for t in texts:
            text = t['text'].replace(' ', '')

            # Ищем 12 цифр
            match = re.search(r'(\d{12})', text)
            if match:
                sku = match.group(1)
                # SKU обычно не начинается с 46
                if not sku.startswith('46'):
                    return sku

        return ''

    def _extract_discount(self, texts: List[Dict]) -> str:
        """Извлечение скидки."""
        for t in texts:
            text = t['text']

            # Ищем проценты
            match = re.search(r'-?(\d+)%', text)
            if match:
                return f"-{match.group(1)}%"

            # Или слова со скидкой
            if any(word in text.lower() for word in ['скидка', 'акция', 'выгода']):
                return text

        return 'нет'


def load_ocr_processor(use_gpu: bool = True) -> OCRProcessor:
    """Фабричная функция для загрузки OCR."""
    # Путь к обученной модели
    MODELS_DIR = Path(__file__).parent.parent.parent / "models"
    paddleocr_dir = Path(__file__).parent.parent.parent / "PaddleOCR"
    custom_rec_dir = str(MODELS_DIR / "custom_ocr")
    custom_dict_path = paddleocr_dir / "ppocr" / "utils" / "ppocr_keys_v1.txt"
    
    # Проверяем есть ли обученная модель (нужны оба файла)
    has_params = (MODELS_DIR / "custom_ocr" / "inference.pdiparams").exists()
    has_model = (MODELS_DIR / "custom_ocr" / "inference.pdmodel").exists()
    
    if has_params and has_model:
        print(f"[OCR] Using trained model: {custom_rec_dir}")
        rec_dict = str(custom_dict_path) if custom_dict_path.exists() else None
        return OCRProcessor(
            use_gpu=use_gpu,
            rec_model_dir=custom_rec_dir,
            rec_char_dict_path=rec_dict
        )
    else:
        print("[OCR] Using default PP-OCRv4 model")
        return OCRProcessor(use_gpu=use_gpu)


def post_process_text(text: str, field_type: str) -> str:
    """Пост-обработка распознанного текста."""
    if not text:
        return ''

    # Убираем лишние пробелы
    text = ' '.join(text.split())

    if field_type == 'barcode':
        # Оставляем только цифры
        text = re.sub(r'\D', '', text)
    elif field_type == 'price':
        # Нормализация цены
        text = text.replace(',', '.').replace(' ', '')
        # Проверяем формат
        if re.match(r'^\d+[.]\d{2}$', text):
            return text
        return ''

    return text
