import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TMP_DIR = BASE_DIR / "tmp"
JOBS_DIR = TMP_DIR / "jobs"

TMP_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

PROGRESS_UPDATE_INTERVAL = 0.5  
MOCK_PROCESSING_TIME = 10  

# CSV колонки согласно заданию (30 полей)
CSV_COLUMNS = [
    # Данные с ценника
    'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
    'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
    'additional_info', 'color', 'special_symbols',
    'frame_timestamp', 'x_min', 'y_min', 'x_max', 'y_max',
    # Данные из QR
    'qr_code_barcode', 'price1_qr', 'price2_qr', 'price3_qr', 'price4_qr',
    'wholesale_level_1_count', 'wholesale_level_1_price',
    'wholesale_level_2_count', 'wholesale_level_2_price',
    'action_price_qr', 'action_code_qr'
]

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255) 
TEXT_THICKNESS = 2
