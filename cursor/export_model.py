#!/usr/bin/env python3
"""Экспорт модели"""
import subprocess
from pathlib import Path

BASE = Path("e:/brbrbr/lent4/backend/ml")
PYTHON = Path("e:/brbrbr/lent4/venv/Scripts/python.exe")

cmd = [
    str(PYTHON),
    str(BASE / "PaddleOCR" / "tools" / "export_model.py"),
    "-c", str(BASE / "PaddleOCR" / "configs" / "rec" / "ocr_lenta.yml"),
    "-o",
    f"Global.pretrained_model={BASE / 'PaddleOCR' / 'output' / 'rec' / 'ocr_lenta' / 'best_accuracy'}",
    f"Global.save_inference_dir={BASE / 'models' / 'custom_ocr'}"
]

print("Запускаю экспорт...")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
print(f"Return code: {result.returncode}")
