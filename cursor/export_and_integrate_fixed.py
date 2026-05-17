#!/usr/bin/env python3
"""
Шаг 6: Экспорт обученной OCR модели и интеграция
"""

import os
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path("e:/brbrbr/lent4/backend/ml")
PADDLEOCR_DIR = BASE_DIR / "PaddleOCR"
MODELS_DIR = BASE_DIR / "models"

def export_model():
    """Экспорт модели в формат inference"""
    print("=== ЭКСПОРТ МОДЕЛИ ===")
    
    # Путь к лучшей модели
    checkpoint_dir = PADDLEOCR_DIR / "output" / "rec" / "ocr_lenta"
    best_model = checkpoint_dir / "best_accuracy.pdparams"
    
    if not best_model.exists():
        print(f"Ошибка: {best_model} не найден!")
        return False
    
    print(f"Найдена модель: {best_model}")
    
    config_path = PADDLEOCR_DIR / "configs" / "rec" / "ocr_lenta.yml"
    save_dir = MODELS_DIR / "custom_ocr"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    export_script = PADDLEOCR_DIR / "tools" / "export_model.py"
    
    # Команда экспорта
    cmd = [
        str(BASE_DIR.parent.parent.parent / "venv" / "Scripts" / "python.exe"),
        str(export_script),
        "-c", str(config_path),
        "-o", 
        f"Global.pretrained_model={best_model}",
        f"Global.save_inference_dir={save_dir}"
    ]
    
    print(f"\nЗапускаю экспорт...")
    print(f"Команда: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PADDLEOCR_DIR))
    
    if result.returncode == 0:
        print("✓ Экспорт завершен успешно!")
        return True
    else:
        print(f"✗ Ошибка экспорта:")
        print(result.stderr[:500])
        return False

def integrate_model():
    """Обновить ocr_processor.py для использования обученной модели"""
    print("\n=== ИНТЕГРАЦИЯ МОДЕЛИ ===")
    
    ocr_processor_path = BASE_DIR / "src" / "ocr" / "ocr_processor.py"
    
    if not ocr_processor_path.exists():
        print(f"Ошибка: {ocr_processor_path} не найден!")
        return False
    
    # Читаем текущий файл
    with open(ocr_processor_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаем бэкап
    backup_path = str(ocr_processor_path) + '.backup'
    shutil.copy2(ocr_processor_path, backup_path)
    print(f"✓ Бэкап создан")
    
    # Заменяем путь к модели на нашу custom_ocr
    if 'custom_ocr' in content:
        print("Модель уже интегрирована!")
        return True
    
    # Простая замена - заменяем MODELS_DIR на MODELS_DIR + custom_ocr
    # Ищем строки с rec_model_dir
    new_content = content.replace(
        '"rec_model_dir": MODELS_DIR',
        '"rec_model_dir": os.path.join(MODELS_DIR, "custom_ocr")'
    )
    
    if new_content == content:
        # Пробуем другой паттерн
        new_content = content.replace(
            "'rec_model_dir': MODELS_DIR",
            "'rec_model_dir': os.path.join(MODELS_DIR, 'custom_ocr')"
        )
    
    with open(ocr_processor_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✓ ocr_processor.py обновлен!")
    return True

def verify_integration():
    """Проверить, что модель готова к использованию"""
    print("\n=== ПРОВЕРКА ===")
    
    model_dir = MODELS_DIR / "custom_ocr"
    
    if not model_dir.exists():
        print(f"✗ Директория {model_dir} не найдена!")
        return False
    
    required_files = ['inference.pdiparams', 'inference.pdmodel']
    
    found = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024*1024)
            found.append(f"  ✓ {file} ({size:.1f} MB)")
        else:
            print(f"  ✗ {file} не найден")
    
    for f in found:
        print(f)
    
    if len(found) == len(required_files):
        print("\n✓ Все файлы модели на месте!")
        return True
    
    return False

def main():
    print("="*60)
    print("ШАГ 6: Экспорт и интеграция OCR модели")
    print("="*60)
    
    # Экспорт
    if export_model():
        # Интеграция
        integrate_model()
        # Проверка
        if verify_integration():
            print("\n" + "="*60)
            print("✓ УСПЕХ! Модель готова!")
            print("="*60)
            print("\nТеперь тестируем:")
            print("  .\\venv\\Scripts\\python.exe test_ml.py")
            return True
    
    print("\n✗ Ошибка в процессе")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
