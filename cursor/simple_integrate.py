#!/usr/bin/env python3
"""Простая интеграция - копируем файлы модели"""
import shutil
from pathlib import Path

BASE = Path("e:/brbrbr/lent4/backend/ml")
src = BASE / "PaddleOCR" / "output" / "rec" / "ocr_lenta"
dst = BASE / "models" / "custom_ocr"

# Создаем директорию
dst.mkdir(parents=True, exist_ok=True)

# Копируем best модель
files_to_copy = [
    ("best_accuracy.pdparams", "inference.pdiparams"),
    ("best_accuracy.pdopt", "inference.pdopt"),
]

for src_name, dst_name in files_to_copy:
    src_file = src / src_name
    dst_file = dst / dst_name
    if src_file.exists():
        shutil.copy2(src_file, dst_file)
        size = dst_file.stat().st_size / (1024*1024)
        print(f"OK: {src_name} -> {dst_name} ({size:.1f} MB)")
    else:
        print(f"MISSING: {src_name}")

# Проверяем что скопировалось
print(f"\nФайлы в {dst}:")
for f in dst.iterdir():
    print(f"  - {f.name}")

# Теперь обновляем ocr_processor.py
ocr_path = BASE / "src" / "ocr" / "ocr_processor.py"

with open(ocr_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Заменяем MODELS_DIR на путь к custom_ocr
if 'custom_ocr' not in content:
    # Делаем бэкап
    shutil.copy2(ocr_path, str(ocr_path) + '.original')
    
    # Заменяем
    new_content = content.replace(
        '"rec_model_dir": MODELS_DIR',
        '"rec_model_dir": os.path.join(MODELS_DIR, "custom_ocr")'
    )
    
    with open(ocr_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("\nOK: ocr_processor.py обновлен для custom_ocr")
else:
    print("\nOK: ocr_processor.py уже настроен на custom_ocr")

print("\nГотово! Теперь можно тестировать:")
print("  .\\venv\\Scripts\\python.exe test_ml.py")
