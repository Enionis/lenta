#!/usr/bin/env python3
"""Ручной экспорт модели через paddle"""
import sys
sys.path.insert(0, 'backend/ml/PaddleOCR')

import paddle
from pathlib import Path

BASE = Path("e:/brbrbr/lent4/backend/ml")
checkpoint_path = BASE / "PaddleOCR/output/rec/ocr_lenta/best_accuracy"
save_dir = BASE / "models/custom_ocr"

print(f"Loading checkpoint from: {checkpoint_path}")

# Загружаем checkpoint
paddle.set_device('cpu')

# Загружаем параметры
state_dict = paddle.load(str(checkpoint_path) + ".pdparams")

print(f"Loaded {len(state_dict)} parameters")

# Сохраняем в формате inference
save_dir.mkdir(parents=True, exist_ok=True)

# Сохраняем параметры
paddle.save(state_dict, str(save_dir / "inference.pdiparams"))

print(f"Model saved to: {save_dir}")
print(f"Files: {list(save_dir.iterdir())}")
