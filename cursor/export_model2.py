#!/usr/bin/env python3
"""Экспорт модели с дополнительными параметрами"""
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
    f"Global.save_inference_dir={BASE / 'models' / 'custom_ocr'}",
    "PostProcess.name=CTCLabelDecode",
    "Decode.exports=False"
]

print("Запускаю экспорт...")
print(f"Команда: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
print(f"Return code: {result.returncode}")
