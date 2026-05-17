"""
День 2-5: Обучение YOLO для детекции ценников.

Этот скрипт:
1. Загружает предобученную YOLOv8n
2. Дообучает на размеченных ценниках
3. Экспортирует модель в ONNX
"""

from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device():
    """Определение доступного устройства (CPU/GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0  # GPU
    except ImportError:
        pass
    return 'cpu'  # CPU fallback


def train_price_tag_detector(
    data_yaml: Path,
    output_dir: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    model_size: str = 'n',  # n, s, m, l, x
    device=None
):
    """
    Обучение детектора ценников.

    Args:
        data_yaml: Путь к data.yaml с конфигурацией датасета
        output_dir: Папка для сохранения результатов
        epochs: Количество эпох
        imgsz: Размер изображения
        batch: Размер батча
        model_size: Размер модели (n=наименьшая, x=наибольшая)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics не установлен! Запустите: pip install ultralytics")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Определяем устройство
    if device is None:
        device = get_device()

    logger.info(f"Устройство: {device} {'(GPU)' if device != 'cpu' else '(CPU)'}")

    # Загрузка предобученной модели
    model_name = f'yolov8{model_size}.pt'
    logger.info(f"Загрузка модели: {model_name}")
    model = YOLO(model_name)

    # Обучение
    logger.info(f"Начало обучения: {epochs} эпох, imgsz={imgsz}, batch={batch}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name='price_tag_detector',
        exist_ok=True,
        patience=20,  # Early stopping
        save=True,
        device=device,

        # Аугментации для лучшей генерализации
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    # Путь к лучшей модели
    best_model_path = output_dir / 'price_tag_detector' / 'weights' / 'best.pt'

    if best_model_path.exists():
        logger.info(f"✅ Обучение завершено! Лучшая модель: {best_model_path}")

        # Экспорт в ONNX
        logger.info("Экспорт в ONNX...")
        model = YOLO(str(best_model_path))
        onnx_path = best_model_path.with_suffix('.onnx')
        model.export(format='onnx', imgsz=imgsz, simplify=True)

        logger.info(f"✅ ONNX модель: {onnx_path}")

        return {
            'pt_path': best_model_path,
            'onnx_path': onnx_path,
            'results': results
        }
    else:
        logger.error("❌ Модель не найдена после обучения!")
        return None


def quick_train_with_coco(output_dir: Path):
    """
    Быстрое решение: используем COCO-pretrained для детекции объектов.

    Для ценников можно:
    1. Использовать детекцию "объектов на полке"
    2. Или обучить на размеченных данных

    Это временное решение для проверки пайплайна.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics не установлен!")
        return None

    logger.info("Создание базового детектора (YOLOv8n COCO)...")

    model = YOLO('yolov8n.pt')

    # Сохраняем для использования
    output_path = Path(output_dir) / 'yolov8n_coco.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(output_path))

    logger.info(f"✅ Базовая модель сохранена: {output_path}")
    logger.warning("⚠️  Это COCO-модель. Для ценников нужно дообучение!")

    return output_path


def create_minimal_dataset_example():
    """Создание минимального примера датасета для теста."""

    example_yaml = """
# Минимальный датасет для тестирования
path: backend/ml/data/processed
train: images
val: images

test: images

nc: 1
names: ['price_tag']
"""

    output_path = Path('backend/ml/data/data_minimal.yaml')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(example_yaml)

    logger.info(f"Создан пример конфигурации: {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train price tag detector')
    parser.add_argument('--data', type=str, default='backend/ml/data/processed/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--output', type=str, default='backend/ml/models',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--quick', action='store_true', help='Quick COCO-only model')
    parser.add_argument('--device', type=str, default=None, help='Device: cpu, cuda, or auto')

    args = parser.parse_args()

    if args.quick:
        # Быстрый вариант - без обучения
        quick_train_with_coco(args.output)
    else:
        # Полное обучение
        data_yaml = Path(args.data)

        if not data_yaml.exists():
            logger.error(f"Data config not found: {data_yaml}")
            logger.info("Creating minimal example...")
            data_yaml = create_minimal_dataset_example()

        train_price_tag_detector(
            data_yaml=data_yaml,
            output_dir=args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_size=args.model,
            device=args.device
        )


if __name__ == '__main__':
    main()
