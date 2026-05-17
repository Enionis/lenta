"""Post-processing for OCR text recognition."""

import re
from typing import Dict, List, Optional


class OCRPostProcessor:
    """Пост-обработка распознанного текста для ценников Ленты."""

    # Коррекция распространенных ошибок OCR
    OCR_CORRECTIONS = {
        # Русские буквы
        '0': 'О',  # Ноль вместо О
        '3': 'З',  # Три вместо З
        '6': 'б',  # Шесть вместо б
        '1': 'л',  # Один вместо л
        'A': 'А',  # Латинская А в русскую
        'B': 'В',
        'C': 'С',
        'E': 'Е',
        'H': 'Н',
        'K': 'К',
        'M': 'М',
        'O': 'О',
        'P': 'Р',
        'T': 'Т',
        'X': 'Х',
        'Y': 'У',
    }

    # Дополнительная нормализация похожих символов для product_name
    CYR_SIMILAR_MAP = {
        'A': 'А', 'a': 'а',
        'B': 'В', 'E': 'Е', 'e': 'е',
        'K': 'К', 'k': 'к',
        'M': 'М', 'H': 'Н',
        'O': 'О', 'o': 'о',
        'P': 'Р', 'p': 'р',
        'C': 'С', 'c': 'с',
        'T': 'Т', 'X': 'Х',
        'Y': 'У', 'y': 'у',
    }

    # Шаблоны полей ценника
    PATTERNS = {
        'barcode': re.compile(r'(\d{13})'),  # EAN13
        'sku': re.compile(r'(\d{12,14})'),  # Артикул
        'price': re.compile(r'(\d+[,.]\d{2})'),  # Цена
        'discount': re.compile(r'-(\d+)%|(\d+)%'),  # Скидка
        'date': re.compile(r'(\d{2}[.]\d{2}[.]\d{4})'),  # Дата
        'code': re.compile(r'(\d{2}_\d+(_\d+)*)'),  # Код зоны
    }

    NAME_STOPWORDS = {
        'нет', 'цена', 'карт', 'карте', 'карта', 'руб', 'шт', 'скид',
        'barcode', 'ean', 'qr', 'код', 'артикул'
    }

    def __init__(self):
        pass

    def correct_text(self, text: str, field_type: str = 'general') -> str:
        """
        Коррекция текста от ошибок OCR.

        Args:
            text: Исходный текст
            field_type: Тип поля (barcode, price, product_name, etc.)

        Returns:
            Исправленный текст
        """
        if not text:
            return ''

        text = text.strip()

        if field_type == 'barcode':
            # Оставляем только цифры
            return re.sub(r'\D', '', text)

        elif field_type == 'price':
            # Нормализация цены
            text = text.replace(',', '.').replace(' ', '')
            # Ищем число формата XXX.XX
            match = re.search(r'(\d+[.]\d{2})', text)
            return match.group(1) if match else ''

        elif field_type == 'product_name':
            # Коррекция букв
            corrected = []
            for char in text:
                if char in self.OCR_CORRECTIONS:
                    corrected.append(self.OCR_CORRECTIONS[char])
                else:
                    corrected.append(char)
            return ''.join(corrected)

        return text

    def extract_fields(self, texts: List[str], text_items: List[Dict] = None) -> Dict:
        """
        Извлечение структурированных полей из списка текстов.

        Args:
            texts: Список распознанных строк

        Returns:
            Словарь с полями
        """
        result = {
            'product_name': '',
            'price_default': '',
            'price_card': '',
            'barcode': '',
            'id_sku': '',
            'discount_amount': '',
            'print_datetime': '',
            'code': '',
            'special_symbols': ''
        }

        all_text = ' '.join(texts)

        # Извлекаем штрихкод (EAN-13) из "грязного" OCR текста.
        result['barcode'] = self._extract_barcode_robust(texts)

        # Извлекаем артикул (12 цифр, не штрихкод)
        for text in texts:
            numbers = re.findall(r'\d{12,14}', text)
            for num in numbers:
                if len(num) == 12 and not num.startswith('46') and num != result['barcode']:
                    result['id_sku'] = num
                    break
            if result['id_sku']:
                break

        # Извлекаем цены (с ранжированием)
        prices = self._extract_prices_robust(texts)
        if prices:
            p_default = prices[0]
            result['price_default'] = str(p_default)

            # Ищем цену по карте среди более низких цен.
            card_candidates = [p for p in prices[1:] if p < (p_default - 0.5)]
            if card_candidates:
                # Цена по карте обычно не экстремально мала относительно обычной.
                card_candidates = [p for p in card_candidates if p >= p_default * 0.30]
                discount_pct = self._extract_discount_percent(texts)
                if discount_pct is not None and 1 <= discount_pct <= 90:
                    target = p_default * (1.0 - discount_pct / 100.0)
                    if card_candidates:
                        best = min(card_candidates, key=lambda p: abs(p - target))
                        result['price_card'] = str(best)
                else:
                    # Обычно цена по карте - минимальная из публичных ценников.
                    if card_candidates:
                        result['price_card'] = str(min(card_candidates))

        # Вычисляем скидку
        if result['price_default'] and result['price_card']:
            try:
                p_def = float(result['price_default'])
                p_card = float(result['price_card'])
                if p_def > p_card:
                    discount = round((1 - p_card / p_def) * 100)
                    result['discount_amount'] = f"-{discount}%"
            except:
                pass

        # Ищем дату
        for text in texts:
            date_match = self.PATTERNS['date'].search(text)
            if date_match:
                result['print_datetime'] = date_match.group(1)
                break

        # Ищем код зоны
        for text in texts:
            code_match = self.PATTERNS['code'].search(text)
            if code_match:
                result['code'] = code_match.group(1)
                break

        # Ищем специальные символы
        for text in texts:
            if text.strip() in ['К', 'Ш', 'к', 'ш']:
                result['special_symbols'] = text.strip().upper()
                break
            elif 'коробка' in text.lower() or 'кейс' in text.lower():
                result['special_symbols'] = 'К'
                break
            elif 'штука' in text.lower() or 'шт.' in text.lower():
                result['special_symbols'] = 'Ш'
                break

        # Название продукта (с учетом confidence и состава строки)
        result['product_name'] = self._extract_product_name_robust(texts, text_items)

        return result

    def _extract_prices_robust(self, texts: List[str]) -> List[float]:
        prices = []
        for text in texts:
            raw = (text or "").replace(" ", "")
            # Стандартный формат XXX.XX / XXX,XX
            found = re.findall(r'(\d+)[,.](\d{2})', raw)
            for p in found:
                try:
                    value = float(f"{p[0]}.{p[1]}")
                    # Отсекаем заведомо шумные OCR-цены.
                    if 15 <= value <= 100000:
                        prices.append(value)
                except Exception:
                    pass
        # Убираем дубли с грубым округлением (OCR часто повторяет ту же цену)
        uniq = sorted(set(round(v, 2) for v in prices), reverse=True)
        return uniq

    def _extract_product_name_robust(self, texts: List[str], text_items: List[Dict] = None) -> str:
        # Если нет text_items, строим заглушки с confidence=0
        items = text_items or [{'text': t, 'confidence': 0.0} for t in texts]
        candidates = []

        for item in items:
            text = self._normalize_product_text(str(item.get('text', '')).strip())
            conf = float(item.get('confidence', 0.0) or 0.0)
            if conf < 0.25 and len(text) < 15:
                continue
            if len(text) < 6 or len(text) > 120:
                continue
            # Отбрасываем тех. строки
            if re.search(r'\d{13}', text):
                continue
            if re.search(r'\d{2}[.]\d{2}[.]\d{4}', text):
                continue
            if re.search(r'(\d+)[,.](\d{2})', text):
                continue
            if '%' in text:
                continue
            if re.match(r'^[\d\s,.;:_-]+$', text):
                continue

            letters = re.findall(r'[A-Za-zА-Яа-яЁё]', text)
            cyr_letters = re.findall(r'[А-Яа-яЁё]', text)
            digits = re.findall(r'\d', text)
            alpha_ratio = len(letters) / max(1, len(text))
            cyr_ratio = len(cyr_letters) / max(1, len(letters))
            digit_ratio = len(digits) / max(1, len(text))
            if alpha_ratio < 0.45 or digit_ratio > 0.20:
                continue
            # Название обычно не полностью латинское шумовое.
            if cyr_ratio < 0.15 and len(text.split()) < 3:
                continue

            # Предпочтительно >=2 слова, но не отбрасываем одиночные
            # длинные кандидаты (иначе теряем полезные строки).
            words = [w for w in text.split() if len(w) >= 2]
            if len(words) < 2 and len(text) < 10:
                continue
            if self._contains_name_stopwords(text):
                continue

            # Скоринг: текстовость + длина + confidence
            word_bonus = 0.3 if len(words) >= 2 else 0.0
            y_bonus = self._name_top_bonus(item.get('bbox'))
            score = alpha_ratio * 2.0 + min(len(text), 40) / 40.0 + conf + cyr_ratio + word_bonus + y_bonus
            candidates.append((score, text))

        if not candidates:
            # Мягкий fallback: берем самую длинную строку,
            # которая не похожа на цену/barcode/дату.
            fallback = []
            for item in items:
                raw_text = str(item.get('text', '')).strip()
                text = self._normalize_product_text(raw_text)
                if len(text) < 8:
                    continue
                if re.search(r'\d{13}', text):
                    continue
                if re.search(r'\d{2}[.]\d{2}[.]\d{4}', text):
                    continue
                if re.search(r'(\d+)[,.](\d{2})', text):
                    continue
                if re.match(r'^[\d\s,.;:_-]+$', text):
                    continue
                fallback.append(text)

            return max(fallback, key=len) if fallback else ''

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _normalize_product_text(self, text: str) -> str:
        """
        Нормализация OCR-строки под product_name:
        - удаление мусорных символов,
        - замена латиницы, визуально похожей на кириллицу,
        - сжатие пробелов.
        """
        if not text:
            return ''

        # Замена похожих символов
        mapped = ''.join(self.CYR_SIMILAR_MAP.get(ch, ch) for ch in text)

        # Оставляем буквы/цифры/базовую пунктуацию и пробел
        mapped = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s\-\.,]", " ", mapped)
        # Убираем одиночные "мусорные" токены вроде 'x' между словами.
        mapped = re.sub(r"\b[^\W\d_]{1}\b", " ", mapped, flags=re.UNICODE)
        mapped = re.sub(r"\s+", " ", mapped).strip()
        return mapped

    def _extract_discount_percent(self, texts: List[str]) -> Optional[int]:
        for text in texts:
            for match in re.findall(r'-(\d+)%|(\d+)%', text or ''):
                raw = match[0] or match[1]
                try:
                    value = int(raw)
                    if 1 <= value <= 90:
                        return value
                except Exception:
                    continue
        return None

    def _contains_name_stopwords(self, text: str) -> bool:
        lower = text.lower()
        for token in self.NAME_STOPWORDS:
            if len(token) <= 3:
                if re.search(rf'\b{re.escape(token)}\b', lower):
                    return True
            elif token in lower:
                return True
        return False

    def _name_top_bonus(self, bbox) -> float:
        """
        Бонус для строк из верхней части ROI (где обычно product_name).
        bbox может быть polygon [[x,y], ...] или прямоугольник [x1,y1,x2,y2].
        """
        if not bbox:
            return 0.0
        try:
            ys = []
            if isinstance(bbox, (list, tuple)):
                for point in bbox:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        ys.append(float(point[1]))
                if not ys and len(bbox) >= 4 and all(isinstance(v, (int, float)) for v in bbox[:4]):
                    ys.extend([float(bbox[1]), float(bbox[3])])
            if not ys:
                return 0.0
            y_min = min(ys)
            # Чем выше строка, тем больше бонус.
            return 0.4 if y_min < 80 else (0.2 if y_min < 150 else 0.0)
        except Exception:
            return 0.0

    def _extract_barcode_robust(self, texts: List[str]) -> str:
        """
        Робастное извлечение barcode из OCR:
        - убираем все нецифровые символы,
        - ищем 13-значные окна,
        - приоритет префиксу 46,
        - проверяем checksum EAN-13.
        """
        candidates = []

        # Кандидаты из каждой строки
        for text in texts:
            digits = re.sub(r"\D", "", text or "")
            if len(digits) < 13:
                continue
            for i in range(0, len(digits) - 12):
                candidates.append(digits[i:i + 13])

        # Кандидаты из всех строк вместе (бывает, что OCR "рвет" barcode)
        all_digits = re.sub(r"\D", "", " ".join(texts))
        if len(all_digits) >= 13:
            for i in range(0, len(all_digits) - 12):
                candidates.append(all_digits[i:i + 13])

        if not candidates:
            return ""

        # 1) Валидный EAN13 + startswith(46)
        for code in candidates:
            if code.startswith("46") and self._is_valid_ean13(code):
                return code

        # 2) Любой валидный EAN13
        for code in candidates:
            if self._is_valid_ean13(code):
                return code

        # 3) startswith(46), даже если checksum не прошел
        for code in candidates:
            if code.startswith("46"):
                return code

        # 4) fallback
        return candidates[0]

    def _is_valid_ean13(self, code: str) -> bool:
        """Проверка контрольной суммы EAN-13."""
        if not code or len(code) != 13 or not code.isdigit():
            return False

        digits = [int(ch) for ch in code]
        checksum = digits[-1]
        body = digits[:-1]

        # Для 12 цифр: сумма нечетных + 3*сумма четных (по позициям с 1)
        odd_sum = sum(body[0::2])
        even_sum = sum(body[1::2])
        calc = (10 - ((odd_sum + 3 * even_sum) % 10)) % 10
        return calc == checksum

    def merge_with_qr(self, ocr_data: Dict, qr_data: Dict) -> Dict:
        """
        Объединение данных OCR и QR.

        QR данные имеют приоритет для barcode и цен.
        """
        result = ocr_data.copy()

        # QR имеет приоритет для штрихкода
        if qr_data.get('qr_code_barcode'):
            result['barcode'] = qr_data['qr_code_barcode']

        # QR цены имеют приоритет
        if qr_data.get('price1_qr'):
            result['price_default'] = qr_data['price1_qr']
        if qr_data.get('price4_qr'):
            result['price_card'] = qr_data['price4_qr']

        return result


def post_process_ocr_results(ocr_data: Dict, qr_data: Dict = None) -> Dict:
    """Удобная функция для пост-обработки."""
    processor = OCRPostProcessor()

    # Коррекция текстов
    for key in ['product_name', 'barcode', 'id_sku']:
        if key in ocr_data:
            ocr_data[key] = processor.correct_text(ocr_data[key], key)

    # Объединение с QR если есть
    if qr_data:
        ocr_data = processor.merge_with_qr(ocr_data, qr_data)

    return ocr_data
