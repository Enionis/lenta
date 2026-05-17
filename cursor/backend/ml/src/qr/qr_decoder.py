"""QR/Barcode decoding using pyzbar and OpenCV."""

import re
import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import cv2

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logging.warning("pyzbar not available. QR decoding will not work.")

logger = logging.getLogger(__name__)


class QRDecoder:
    """Декодер QR кодов и штрихкодов с ценников Ленты."""

    def __init__(self):
        if not PYZBAR_AVAILABLE:
            raise RuntimeError("pyzbar not installed. Run: pip install pyzbar")

        self.symbols = [
            ZBarSymbol.QRCODE,
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39
        ]
        self.barcode_symbols = [
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
        ]

    def decode(self, image: np.ndarray) -> List[Dict]:
        """
        Декодирование всех QR/штрихкодов на изображении.

        Args:
            image: numpy array (BGR)

        Returns:
            Список словарей с данными:
            [{
                'data': 'распознанные данные',
                'type': 'QRCODE' или 'EAN13' и т.д.,
                'rect': (x, y, w, h)
            }, ...]
        """
        if image is None or image.size == 0:
            return []

        # Конвертация в grayscale для лучшего распознавания
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Multi-pass декодирование для сложных/размытых штрихкодов.
        variants = self._generate_variants(gray)
        results = []
        seen = set()

        for variant in variants:
            decoded = pyzbar.decode(variant, symbols=self.symbols)
            for item in self._format_results(decoded):
                key = (item.get('type', ''), item.get('data', ''))
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)

        # Fallback: пробуем OpenCV QRCodeDetector для QR (если pyzbar не дал ничего)
        if not results:
            results.extend(self._decode_qr_opencv(gray))

        # Дополнительный проход: ищем квадратные QR-кандидаты и декодируем их отдельно.
        if not results:
            candidate_crops = self._build_qr_candidate_crops(gray)
            for crop in candidate_crops:
                if crop is None or crop.size == 0:
                    continue

                # pyzbar на вариантах crop
                for variant in self._generate_variants(crop):
                    decoded = pyzbar.decode(variant, symbols=self.symbols)
                    for item in self._format_results(decoded):
                        key = (item.get('type', ''), item.get('data', ''))
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append(item)

                # OpenCV QR fallback для crop
                if not results:
                    for item in self._decode_qr_opencv(crop):
                        key = (item.get('type', ''), item.get('data', ''))
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append(item)
                if results:
                    break

        return results

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Предобработка изображения для улучшения QR."""
        # Увеличение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Небольшое размытие для уменьшения шума
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return blurred

    def _generate_variants(self, gray: np.ndarray) -> List[np.ndarray]:
        """Генерация вариантов изображения для устойчивого decode."""
        variants = [gray]
        preprocessed = self._preprocess(gray)
        variants.append(preprocessed)

        # Увеличение
        upscaled = cv2.resize(preprocessed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)
        upscaled3 = cv2.resize(preprocessed, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled3)

        # Бинаризация Otsu
        _, otsu = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        variants.append(adaptive)

        # Инвертированная бинаризация
        variants.append(cv2.bitwise_not(otsu))

        # Легкая резкость
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(preprocessed, -1, kernel)
        variants.append(sharpened)

        # Повороты (часто QR/штрихкод с наклоном)
        for base in [gray, preprocessed, upscaled]:
            if base is None or base.size == 0:
                continue
            variants.append(cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE))
            variants.append(cv2.rotate(base, cv2.ROTATE_180))
            variants.append(cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE))

        return variants

    def _build_qr_candidate_crops(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Поиск вероятных QR-областей по квадратным контурам.
        """
        if gray is None or gray.size == 0:
            return []

        h, w = gray.shape[:2]
        if h < 40 or w < 40:
            return [gray]

        # Подчеркиваем контуры QR
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        edges = cv2.Canny(thr, 80, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        min_area = max(100, int(h * w * 0.0025))
        max_area = int(h * w * 0.35)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) != 4:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / float(bh)
            if not (0.6 <= aspect <= 1.4):
                continue

            boxes.append((x, y, bw, bh))

        # dedupe похожих боксов (очень грубо по центру)
        dedup = []
        for b in boxes:
            x, y, bw, bh = b
            cx, cy = x + bw // 2, y + bh // 2
            if any(abs(cx - (dx + dw // 2)) < 10 and abs(cy - (dy + dh // 2)) < 10 for dx, dy, dw, dh in dedup):
                continue
            dedup.append(b)

        # Кропы с небольшим контекстом
        crops = []
        for x, y, bw, bh in dedup[:15]:
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            if x2 > x1 and y2 > y1:
                crop = gray[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

        return crops

    def _decode_qr_opencv(self, gray: np.ndarray) -> List[Dict]:
        """Fallback QR decode через OpenCV."""
        results = []
        try:
            detector = cv2.QRCodeDetector()
            # 1) Multi decode
            retval, decoded_infos, points, _ = detector.detectAndDecodeMulti(gray)
            if retval and decoded_infos is not None:
                for i, data in enumerate(decoded_infos):
                    if not data:
                        continue
                    rect = (0, 0, gray.shape[1], gray.shape[0])
                    if points is not None and i < len(points):
                        pts = points[i]
                        x = int(np.min(pts[:, 0]))
                        y = int(np.min(pts[:, 1]))
                        w = int(np.max(pts[:, 0]) - x)
                        h = int(np.max(pts[:, 1]) - y)
                        rect = (x, y, w, h)
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'rect': rect,
                        'polygon': None
                    })
            # 2) Single decode fallback
            if not results:
                data, points, _ = detector.detectAndDecode(gray)
                if data:
                    rect = (0, 0, gray.shape[1], gray.shape[0])
                    if points is not None and len(points) > 0:
                        pts = points
                        x = int(np.min(pts[:, 0]))
                        y = int(np.min(pts[:, 1]))
                        w = int(np.max(pts[:, 0]) - x)
                        h = int(np.max(pts[:, 1]) - y)
                        rect = (x, y, w, h)
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'rect': rect,
                        'polygon': None
                    })
        except Exception:
            pass
        return results

    def _format_results(self, decoded) -> List[Dict]:
        """Форматирование результатов pyzbar."""
        results = []

        for item in decoded:
            data = item.data.decode('utf-8') if isinstance(item.data, bytes) else item.data

            results.append({
                'data': data,
                'type': item.type,
                'rect': (item.rect.left, item.rect.top,
                        item.rect.width, item.rect.height),
                'polygon': item.polygon
            })

        return results

    def parse_lenta_qr(self, data: str) -> Dict:
        """
        Парсинг QR данных ценника Ленты.

        Формат:
        barcode|b;price1|p1;price2|p2;price3|p3;price4|p4;
        wholesaleLevel1Count|wL1C;wholesaleLevel1Price|wL1P;
        wholesaleLevel2Count|wL2C;wholesaleLevel2Price|wL2P;
        actionPrice|aP;actionCode|aC

        Args:
            data: Строка из QR кода

        Returns:
            Словарь с распарсенными полями
        """
        result = {
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

        if not data or ';' not in data:
            # Может быть просто штрихкод
            if data.isdigit() and len(data) == 13:
                result['qr_code_barcode'] = data
            return result

        # Парсим пары в формате:
        # key|alias;value;key|alias;value;...
        parts = [p.strip() for p in data.split(';') if p.strip()]

        i = 0
        while i < len(parts):
            token = parts[i]

            # Формат key|alias + следующий token как value
            if '|' in token and i + 1 < len(parts):
                key_part = token
                value = parts[i + 1]
                key = self._normalize_qr_key(key_part)
                if key in result and value != '':
                    result[key] = value
                i += 2
                continue

            # Формат key:value (сокращенный)
            if ':' in token:
                k, v = token.split(':', 1)
                key = self._normalize_qr_key(k.strip())
                v = v.strip()
                if key in result and v != '':
                    result[key] = v
                i += 1
                continue

            # Может быть просто EAN-13
            if token.isdigit() and len(token) == 13 and not result['qr_code_barcode']:
                result['qr_code_barcode'] = token

            i += 1

        # Альтернативные форматы (дублируем на случай редких строк)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Проверяем сокращенные форматы
            if part.startswith('b:'):
                result['qr_code_barcode'] = part[2:]
            elif part.startswith('p1:'):
                result['price1_qr'] = part[3:]
            elif part.startswith('p2:'):
                result['price2_qr'] = part[3:]
            elif part.startswith('p3:'):
                result['price3_qr'] = part[3:]
            elif part.startswith('p4:'):
                result['price4_qr'] = part[3:]
            elif part.startswith('aP:'):
                result['action_price_qr'] = part[3:]
            elif part.startswith('aC:'):
                result['action_code_qr'] = part[3:]

        return result

    def _normalize_qr_key(self, key_part: str) -> str:
        """Нормализация ключа QR кода."""
        # Отображение псевдонимов на стандартные ключи
        key_map = {
            'barcode': 'qr_code_barcode',
            'b': 'qr_code_barcode',
            'price1': 'price1_qr',
            'p1': 'price1_qr',
            'price2': 'price2_qr',
            'p2': 'price2_qr',
            'price3': 'price3_qr',
            'p3': 'price3_qr',
            'price4': 'price4_qr',
            'p4': 'price4_qr',
            'wholesaleLevel1Count': 'wholesale_level_1_count',
            'wL1C': 'wholesale_level_1_count',
            'wholesaleLevel1Price': 'wholesale_level_1_price',
            'wL1P': 'wholesale_level_1_price',
            'wholesaleLevel2Count': 'wholesale_level_2_count',
            'wL2C': 'wholesale_level_2_count',
            'wholesaleLevel2Price': 'wholesale_level_2_price',
            'wL2P': 'wholesale_level_2_price',
            'actionPrice': 'action_price_qr',
            'aP': 'action_price_qr',
            'actionCode': 'action_code_qr',
            'aC': 'action_code_qr',
        }

        # Убираем значение после | если есть
        if '|' in key_part:
            key = key_part.split('|')[0]
        else:
            key = key_part

        return key_map.get(key, key)

    def extract_barcode(self, image: np.ndarray) -> Optional[str]:
        """Извлечение только штрихкода (EAN13)."""
        results = self.decode(image)

        for r in results:
            if r['type'] in ['EAN13', 'EAN8', 'CODE128']:
                data = r['data']
                # Проверяем формат EAN13 (13 цифр, начинается с 46)
                if data.isdigit() and len(data) == 13 and data.startswith('46'):
                    return data

        return None

    def extract_barcode_1d(self, image: np.ndarray) -> Optional[str]:
        """
        Специализированный 1D-проход для штрихкодов.
        Использует агрессивные варианты preprocessing и только barcode символы.
        """
        if image is None or image.size == 0:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        variants = self._generate_barcode_variants(gray)
        candidates = []
        seen = set()

        for var in variants:
            try:
                decoded = pyzbar.decode(var, symbols=self.barcode_symbols)
            except Exception:
                decoded = []

            for item in decoded:
                data = item.data.decode("utf-8") if isinstance(item.data, bytes) else str(item.data)
                digits = re.sub(r"\D", "", data)
                if len(digits) == 13 and digits not in seen:
                    seen.add(digits)
                    candidates.append(digits)

        # приоритет штрихкодам Ленты (46...) и валидной контрольной сумме
        for code in candidates:
            if code.startswith("46") and self._is_valid_ean13(code):
                return code
        for code in candidates:
            if self._is_valid_ean13(code):
                return code
        for code in candidates:
            if code.startswith("46"):
                return code
        return candidates[0] if candidates else None

    def _generate_barcode_variants(self, gray: np.ndarray) -> List[np.ndarray]:
        """Варианты изображения под 1D barcode decoding."""
        variants = [gray]

        # Легкое выравнивание контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        c = clahe.apply(gray)
        variants.append(c)

        # Увеличение
        up2 = cv2.resize(c, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        up3 = cv2.resize(c, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        variants.extend([up2, up3])

        # Бинаризация
        _, otsu = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))

        # Морфология под вертикальные полосы
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open)
        variants.append(morph)
        variants.append(cv2.bitwise_not(morph))

        # Повороты на случай наклона/вертикальной ориентации
        for base in [gray, c, up2, otsu, morph]:
            if base is None or base.size == 0:
                continue
            variants.append(cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE))
            variants.append(cv2.rotate(base, cv2.ROTATE_180))
            variants.append(cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE))

        return variants

    def _is_valid_ean13(self, code: str) -> bool:
        """Проверка контрольной суммы EAN-13."""
        if not code or len(code) != 13 or not code.isdigit():
            return False
        digits = [int(ch) for ch in code]
        checksum = digits[-1]
        body = digits[:-1]
        odd_sum = sum(body[0::2])
        even_sum = sum(body[1::2])
        calc = (10 - ((odd_sum + 3 * even_sum) % 10)) % 10
        return calc == checksum


def load_qr_decoder() -> QRDecoder:
    """Фабричная функция для загрузки QR декодера."""
    return QRDecoder()
