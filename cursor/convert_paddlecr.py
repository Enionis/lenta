import json
from pathlib import Path
dataset_dir = Path('backend/ml/data/ocr_dataset')
paddle_dir = Path('backend/ml/data/paddleocr_format/train')
paddle_dir.mkdir(parents=True, exist_ok=True)
lines = []
for f in (dataset_dir / 'labels').glob('*.json'):
  with open(f, 'r', encoding='utf-8') as fp:
      data = json.load(fp)
  img_path = (dataset_dir / 'images' / data['image']).absolute()
  if img_path.exists():
      for field in ['product_name', 'price_default', 'barcode', 'id_sku']:
          text = data.get(field, '').strip()
          if text and text.lower() != 'нет':
              lines.append(f'{img_path}\t{text}')
with open(paddle_dir / 'labels.txt', 'w', encoding='utf-8') as f:
  f.write('\n'.join(lines))
print(f'Создано {len(lines)} строк разметки')