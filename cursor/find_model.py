#!/usr/bin/env python3
"""Поиск обученной модели"""
from pathlib import Path

base = Path('e:/brbrbr/lent4')

# Ищем .pdparams
files = list(base.rglob('*.pdparams'))
print(f"Найдено .pdparams файлов: {len(files)}")
for f in files[:20]:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"  {f} ({size_mb:.1f} MB)")

# Ищем директории output
output_dirs = list(base.rglob('output'))
print(f"\nНайдено output директорий: {len(output_dirs)}")
for d in output_dirs:
    print(f"  {d}")
    if d.is_dir():
        try:
            for item in d.iterdir():
                marker = "[DIR]" if item.is_dir() else "[FILE]"
                print(f"    {marker} {item.name}")
        except Exception as e:
            print(f"    Ошибка: {e}")

# Ищем логи обучения
print("\n--- Логи обучения ---")
logs = list(base.rglob('train.log')) + list(base.rglob('*.log'))
for log in logs[:10]:
    print(f"  {log}")
