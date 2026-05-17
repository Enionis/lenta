# -*- coding: utf-8 -*-
"""
Полное дообучение OCR для ценников Ленты.

Шаги:
1. Подготовка датасета из всех видео
2. Конвертация в формат PaddleOCR
3. Скачивание предобученной модели
4. Запуск обучения
5. Экспорт inference модели
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def step1_prepare_dataset():
    """Шаг 1: Подготовка датасета из всех видео."""
    
    logger.info("="*60)
    logger.info("ШАГ 1: Подготовка датасета")
    logger.info("="*60)
    
    video_dir = Path('Данные')
    output_dir = Path('backend/ml/data/ocr_dataset')
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    # Очищаем и создаем папки
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все CSV
    csv_files = list(video_dir.glob('**/*.csv'))
    logger.info(f"Найдено CSV файлов: {len(csv_files)}")
    
    total_images = 0
    all_labels = []
    
    for csv_path in csv_files:
        video_name = csv_path.stem
        video_file = None
        
        # Ищем видео
        for ext in ['.mp4', '.mov', '.avi', '.MOV']:
            candidate = csv_path.parent / f"{video_name}{ext}"
            if candidate.exists():
                video_file = candidate
                break
        
        if not video_file:
            logger.warning(f"Видео не найдено: {video_name}")
            continue
        
        logger.info(f"\nОбработка: {video_file.name}")
        
        # Загружаем разметку
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Ошибка чтения CSV: {e}")
            continue
        
        # Преобразуем координаты
        for col in ['x_min', 'y_min', 'x_max', 'y_max']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Открываем видео
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_file}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_images = 0
        
        # Обрабатываем каждую уникальную временную метку
        timestamps = df['frame_timestamp'].unique()
        
        for timestamp in tqdm(timestamps, desc=f"Кадры {video_name}"):
            frame_num = int((timestamp / 1000) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Проверяем качество кадра
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 50:  # Пропускаем размытые кадры
                continue
            
            # Берем все ценники на этом кадре
            frame_data = df[df['frame_timestamp'] == timestamp]
            
            for idx, row in frame_data.iterrows():
                x_min, y_min = int(row['x_min']), int(row['y_min'])
                x_max, y_max = int(row['x_max']), int(row['y_max'])
                
                h, w = frame.shape[:2]
                x_min = max(0, min(x_min, w))
                x_max = max(0, min(x_max, w))
                y_min = max(0, min(y_min, h))
                y_max = max(0, min(y_max, h))
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Вырезаем ценник
                roi = frame[y_min:y_max, x_min:x_max]
                
                if roi.size == 0:
                    continue
                
                # Проверяем качество ROI
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                
                if roi_laplacian < 30:
                    continue
                
                # Увеличиваем разрешение для лучшего OCR
                scale = 2
                roi_resized = cv2.resize(roi, (roi.shape[1]*scale, roi.shape[0]*scale), 
                                        interpolation=cv2.INTER_CUBIC)
                
                # Сохраняем
                img_name = f"{video_name}_{int(timestamp)}_{idx}.jpg"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), roi_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Подготавливаем разметку
                label_data = {
                    'image': img_name,
                    'video': video_name,
                    'timestamp': int(timestamp),
                    'product_name': str(row.get('product_name', '')).strip(),
                    'price_default': str(row.get('price_default', '')).strip(),
                    'price_card': str(row.get('price_card', '')).strip(),
                    'barcode': str(row.get('barcode', '')).strip(),
                    'id_sku': str(row.get('id_sku', '')).strip(),
                    'quality': float(roi_laplacian),
                    'bbox': [x_min, y_min, x_max, y_max]
                }
                
                # Сохраняем JSON
                label_path = labels_dir / f"{img_name.replace('.jpg', '.json')}"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                
                all_labels.append(label_data)
                video_images += 1
        
        cap.release()
        logger.info(f"  Извлечено: {video_images} изображений")
        total_images += video_images
    
    # Сохраняем общую информацию
    dataset_info = {
        'total_images': total_images,
        'labels': all_labels
    }
    
    with open(output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ДАТАСЕТ ГОТОВ: {total_images} изображений")
    logger.info(f"{'='*60}")
    
    return total_images


def step2_create_paddleocr_format():
    """Шаг 2: Конвертация в формат PaddleOCR."""
    
    logger.info("\n" + "="*60)
    logger.info("ШАГ 2: Конвертация в формат PaddleOCR")
    logger.info("="*60)
    
    dataset_dir = Path('backend/ml/data/ocr_dataset')
    output_dir = Path('backend/ml/data/paddleocr_format')
    
    # Создаем структуру
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for d in [train_dir / 'images', train_dir / 'labels', 
              val_dir / 'images', val_dir / 'labels']:
        d.mkdir(parents=True, exist_ok=True)
    
    # Загружаем разметку
    with open(dataset_dir / 'dataset_info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data['labels']
    
    # Разделяем на train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(labels))
    split = int(0.8 * len(indices))
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    logger.info(f"Разделение: train={len(train_indices)}, val={len(val_indices)}")
    
    # Копируем файлы и создаем labels.txt
    src_images = dataset_dir / 'images'
    
    for split_name, indices in [('train', train_indices), ('val', val_indices)]:
        split_dir = output_dir / split_name
        recognition_lines = []
        
        for idx in tqdm(indices, desc=f"Подготовка {split_name}"):
            label = labels[idx]
            img_name = label['image']
            src_path = src_images / img_name
            dst_path = split_dir / 'images' / img_name
            
            if not src_path.exists():
                continue
            
            # Копируем изображение
            shutil.copy2(src_path, dst_path)
            
            # Формируем строки для recognition
            texts = []
            
            if label.get('product_name') and label['product_name'] not in ['', 'нет']:
                texts.append(label['product_name'])
            
            if label.get('price_default') and label['price_default'] not in ['', 'нет']:
                texts.append(label['price_default'])
            
            if label.get('barcode') and label['barcode'] not in ['', 'нет']:
                texts.append(label['barcode'])
            
            # Формат: путь\tтекст
            for text in texts:
                if text.strip():
                    abs_path = dst_path.absolute()
                    recognition_lines.append(f"{abs_path}\t{text}")
        
        # Сохраняем labels.txt
        with open(split_dir / 'labels.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(recognition_lines))
        
        logger.info(f"{split_name}: {len(recognition_lines)} строк разметки")
    
    logger.info(f"\n{'='*60}")
    logger.info("ФОРМАТ PADDLEOCR ГОТОВ")
    logger.info(f"{'='*60}")


def step3_download_pretrained():
    """Шаг 3: Скачивание предобученной модели."""
    
    logger.info("\n" + "="*60)
    logger.info("ШАГ 3: Скачивание предобученной модели")
    logger.info("="*60)
    
    import urllib.request
    import tarfile
    
    model_dir = Path('backend/ml/models/pretrained')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # URL для PP-OCRv4
    url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
    tar_path = model_dir / "ch_PP-OCRv4_rec_train.tar"
    
    if not tar_path.exists():
        logger.info(f"Скачивание: {url}")
        logger.info("Это может занять 5-10 минут...")
        
        try:
            urllib.request.urlretrieve(url, tar_path)
            logger.info("Скачивание завершено")
        except Exception as e:
            logger.error(f"Ошибка скачивания: {e}")
            logger.info("Пробуем альтернативный метод...")
            # Используем wget или curl
            import subprocess
            subprocess.run(['wget', '-O', str(tar_path), url], check=True)
    else:
        logger.info("Модель уже скачана")
    
    # Распаковываем
    extract_dir = model_dir / "ch_PP-OCRv4_rec_train"
    if not extract_dir.exists():
        logger.info("Распаковка...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(model_dir)
        logger.info("Распаковка завершена")
    
    logger.info(f"\n{'='*60}")
    logger.info("ПРЕДОБУЧЕННАЯ МОДЕЛЬ ГОТОВА")
    logger.info(f"{'='*60}")


def step4_create_config():
    """Шаг 4: Создание конфигурации обучения."""
    
    logger.info("\n" + "="*60)
    logger.info("ШАГ 4: Создание конфигурации")
    logger.info("="*60)
    
    config_content = '''
Global:
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/ocr_lenta
  save_epoch_step: 10
  eval_batch_step: [0, 200]
  cal_metric_during_train: true
  pretrained_model: ./models/pretrained/ch_PP-OCRv4_rec_train/best_accuracy.pdparams
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/en/word_1.png
  character_dict_path: ./dict/lenta_dict.txt
  max_text_length: 50
  infer_mode: false
  use_space_char: true
  distributed: false
  save_res_path: ./output/rec/predicts_ppocrv3.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetNew
    scale: 0.95
  Head:
    name: MultiHead
    out_channels_list:
      CTCLabelDecode: 80
      SARLabelDecode: 128
    # SAR Head
    attention:
      mode: stream
      dim: 128
      hidden_size: 128
      attention_mode: pcurrence_wn_generic_att
    # CTC Head
    fc_decay: 0.0004
  Neck:
    name: svtr
    dims: 120
    depth: 2
    hidden_dims: 120
    use_guide: True

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
        use_focal_loss: false
        weight: 1.0
        mask_others: false
    - SARLoss:
        weight: 1.0
        mask_others: false

PostProcess:
  name: CTCLabelDecode
  SARLabelDecode:

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: False

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/paddleocr_format/train
    ext_op_transform_idx: 1
    label_file_list: ["./data/paddleocr_format/train/labels.txt"]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
          max_text_length: 50
      - RecAug:
      - CTCLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
          padding: False
      - KeepKeys:
          keep_keys: ["image", "label", "length"]
  loader:
    shuffle: true
    batch_size_per_card: 32
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/paddleocr_format/val
    label_file_list: ["./data/paddleocr_format/val/labels.txt"]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - CTCLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
          padding: False
      - KeepKeys:
          keep_keys: ["image", "label", "length"]
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 32
    num_workers: 4
'''
    
    config_path = Path('backend/ml/configs/ocr_train.yml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"Конфигурация сохранена: {config_path}")
    
    # Создаем словарь символов
    dict_content = '''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
абвгдеёжзийклмнопрстуфхцчшщъыьэюя
 .,;:!?-_/%()[]<>=+*#@&$
'''
    
    dict_path = Path('backend/ml/dict/lenta_dict.txt')
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(dict_content)
    
    logger.info(f"Словарь сохранен: {dict_path}")


def step5_run_training():
    """Шаг 5: Запуск обучения."""
    
    logger.info("\n" + "="*60)
    logger.info("ШАГ 5: Запуск обучения")
    logger.info("="*60)
    
    # Проверяем что paddleocr установлен
    try:
        import paddleocr
        logger.info("PaddleOCR установлен")
    except ImportError:
        logger.error("PaddleOCR не установлен!")
        logger.info("Установите: pip install paddleocr[train]")
        return
    
    # Проверяем CUDA
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.cuda.device_count() > 0:
            logger.info(f"CUDA доступна: {paddle.cuda.device_count()} GPU")
        else:
            logger.warning("CUDA не доступна, обучение будет на CPU (медленно)")
    except:
        pass
    
    logger.info("\nКоманда для запуска обучения:")
    logger.info("cd backend/ml && python -m paddleocr.tools.train -c configs/ocr_train.yml")
    
    logger.info("\nИЛИ используйте скрипт:")
    logger.info("python run_ocr_training.py")


def main():
    logger.info("="*70)
    logger.info("ПОЛНОЕ ДООБУЧЕНИЕ OCR ДЛЯ ЦЕННИКОВ ЛЕНТЫ")
    logger.info("="*70)
    
    # Последовательно выполняем шаги
    try:
        # Шаг 1: Подготовка датасета
        total_images = step1_prepare_dataset()
        
        if total_images == 0:
            logger.error("Датасет пуст! Проверьте видео и разметку.")
            return
        
        # Шаг 2: Конвертация
        step2_create_paddleocr_format()
        
        # Шаг 3: Скачивание модели
        step3_download_pretrained()
        
        # Шаг 4: Конфигурация
        step4_create_config()
        
        # Шаг 5: Инструкции по обучению
        step5_run_training()
        
        logger.info("\n" + "="*70)
        logger.info("ПОДГОТОВКА ЗАВЕРШЕНА!")
        logger.info("="*70)
        logger.info("\nСледующий шаг: запустите обучение командой:")
        logger.info("cd backend/ml && python -m paddleocr.tools.train -c configs/ocr_train.yml")
        logger.info("\nВремя обучения:")
        logger.info("  - GPU (RTX 2060): 2-4 часа")
        logger.info("  - CPU: 10-20 часов")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)


if __name__ == '__main__':
    main()
