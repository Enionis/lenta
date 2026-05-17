#!/usr/bin/env python3
"""Попытка экспорта с полным trace"""
import sys
sys.path.insert(0, 'backend/ml/PaddleOCR')

import traceback
try:
    from tools.export_model import main
    sys.argv = [
        'export_model.py',
        '-c', 'backend/ml/PaddleOCR/configs/rec/ocr_lenta.yml',
        '-o',
        'Global.pretrained_model=backend/ml/PaddleOCR/output/rec/ocr_lenta/best_accuracy',
        'Global.save_inference_dir=backend/ml/models/custom_ocr'
    ]
    main()
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
