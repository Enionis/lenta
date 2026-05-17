"""ML package for Lenta Tech price tag recognition.

Использование:
    from ml import process_video
    import pandas as pd

    result = process_video(Path('video.mp4'), Path('output/'))
    df = result['dataframe']  # pd.DataFrame с результатами
"""

from src.pipeline import process_video

__version__ = '0.1.0'
__all__ = ['process_video']
