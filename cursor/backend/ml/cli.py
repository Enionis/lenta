"""
CLI для ML pipeline.

Использование:
    python -m ml.cli --video path/to/video.mp4 --output results.csv
    python -m ml.cli --video path/to/video.mp4 --visualize
"""

import argparse
import logging
import sys
from pathlib import Path

# Добавляем путь к ml
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import process_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Price Tag Recognition - ML Pipeline'
    )
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--output', '-o',
        default='results.csv',
        help='Output CSV file (default: results.csv)'
    )
    parser.add_argument(
        '--output-dir', '-d',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--visualize', '-vis',
        action='store_true',
        help='Show visualization'
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info(f"Processing: {video_path}")
    logger.info(f"Output directory: {output_dir}")

    def progress_callback(progress: int, message: str):
        bar = '=' * (progress // 5) + '>' + ' ' * (20 - progress // 5)
        print(f"\r[{bar}] {progress}% {message}", end='', flush=True)
        if progress == 100:
            print()

    try:
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        df = result['dataframe']
        csv_path = result['csv_path']
        preview_path = result['preview_path']
        stats = result['stats']

        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Detections found: {stats.get('detections_found', 0)}")
        logger.info(f"Results: {stats.get('results_count', 0)} rows")
        logger.info(f"Time: {stats.get('processing_time', 0):.1f} seconds")
        logger.info("=" * 60)
        logger.info(f"CSV: {csv_path}")
        logger.info(f"Preview: {preview_path}")

        # Показываем первые несколько строк
        if len(df) > 0:
            print("\nFirst 3 rows:")
            print(df.head(3).to_string())

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
