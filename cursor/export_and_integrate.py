#!/usr/bin/env python3
"""
Экспорт обученной OCR модели в inference-формат и проверка интеграции.
"""

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "backend" / "ml"
PADDLEOCR_DIR = BASE_DIR / "PaddleOCR"
MODELS_DIR = BASE_DIR / "models"
PYTHON_EXE = ROOT_DIR / "venv" / "Scripts" / "python.exe"


def export_model() -> bool:
    """Экспорт best checkpoint в inference.pdmodel/inference.pdiparams."""
    checkpoint_prefix = Path("output/rec/ocr_lenta/best_accuracy")
    train_config = Path("output/rec/ocr_lenta/config.yml")
    save_dir = Path("../models/custom_ocr")

    if not (PADDLEOCR_DIR / f"{checkpoint_prefix}.pdparams").exists():
        print(f"ERROR: checkpoint not found: {PADDLEOCR_DIR / (str(checkpoint_prefix) + '.pdparams')}")
        return False

    if not (PADDLEOCR_DIR / train_config).exists():
        print(f"ERROR: training config not found: {PADDLEOCR_DIR / train_config}")
        return False

    cmd = [
        str(PYTHON_EXE),
        "tools/export_model.py",
        "-c",
        str(train_config),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix}",
        f"Global.save_inference_dir={save_dir}",
        "Global.use_gpu=False",
        "Global.export_with_pir=False",
    ]

    print("Running export command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=PADDLEOCR_DIR, text=True)
    return result.returncode == 0


def verify_model_files() -> bool:
    """Проверка, что файлы inference модели существуют."""
    model_dir = MODELS_DIR / "custom_ocr"
    required = ["inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        print(f"ERROR: missing files in {model_dir}: {missing}")
        return False
    print(f"OK: custom OCR exported to {model_dir}")
    return True


def main() -> int:
    print("=" * 60)
    print("Export trained OCR model")
    print("=" * 60)

    if not export_model():
        print("ERROR: export failed")
        return 1

    if not verify_model_files():
        return 1

    print("DONE: model is ready for pipeline usage.")
    print("Run test: .\\venv\\Scripts\\python.exe test_ml.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
