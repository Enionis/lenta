"""
Тест ML модуля.
"""

import sys
from pathlib import Path

# Добавляем путь к ml
sys.path.insert(0, str(Path('backend/ml').absolute()))

def test_imports():
    """Тест импортов."""
    print("Testing imports...")

    try:
        from src.pipeline import process_video
        print("[OK] pipeline import")
    except Exception as e:
        print(f"[FAIL] pipeline: {e}")
        return False

    try:
        from src.detection.detector import PriceTagDetector
        print("[OK] detector import")
    except Exception as e:
        print(f"[FAIL] detector: {e}")

    try:
        from src.utils.video_utils import get_video_info
        print("[OK] video_utils import")
    except Exception as e:
        print(f"[FAIL] video_utils: {e}")

    return True


def test_detector():
    """Тест детектора (без видео)."""
    print("\nTesting detector...")

    try:
        from src.detection.detector import load_detector

        # Пытаемся загрузить детектор
        detector = load_detector(use_gpu=False)
        print(f"[OK] Detector loaded: {type(detector)}")

        # Тест на пустом изображении
        import numpy as np
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(empty_frame)
        print(f"[OK] Empty frame detection: {len(dets)} detections")

        return True
    except Exception as e:
        print(f"[FAIL] Detector test: {e}")
        return False


def test_pipeline():
    """Тест pipeline (требуется видео)."""
    print("\nTesting pipeline...")

    # Ищем видео
    video_files = list(Path('Данные').glob('**/*.mp4'))
    video_files += list(Path('Данные').glob('**/*.mov'))

    if not video_files:
        print("[SKIP] No video files found for testing")
        return True

    print(f"Found video: {video_files[0]}")

    try:
        from src.pipeline import process_video

        output_dir = Path('test_outputs')
        output_dir.mkdir(exist_ok=True)

        def progress(p, m):
            print(f"  {p}%: {m}")

        result = process_video(
            video_path=video_files[0],
            output_dir=output_dir,
            progress_callback=progress
        )

        print(f"[OK] Pipeline completed")
        print(f"  Tracks: {result['stats'].get('tracks_found', 0)}")
        print(f"  Results: {result['stats'].get('results_count', 0)}")
        print(f"  CSV: {result['csv_path']}")
        print(f"  Preview: {result['preview_path']}")

        return True
    except Exception as e:
        print(f"[FAIL] Pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("ML Module Test")
    print("=" * 60)

    success = True

    success &= test_imports()
    success &= test_detector()
    success &= test_pipeline()

    print("\n" + "=" * 60)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
