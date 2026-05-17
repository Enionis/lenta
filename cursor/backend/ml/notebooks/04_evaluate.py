"""
День 16-17: Оценка точности решения.

Сравнение результатов с Ground Truth (CSV разметкой).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    """Загрузка предсказаний из CSV."""
    df = pd.read_csv(predictions_csv)
    logger.info(f"Loaded predictions: {len(df)} rows")
    return df


def load_ground_truth(gt_csv: Path) -> pd.DataFrame:
    """Загрузка Ground Truth из разметки."""
    df = pd.read_csv(gt_csv)

    # Преобразуем координаты (заменяем запятую на точку)
    for col in ['x_min', 'y_min', 'x_max', 'y_max']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    logger.info(f"Loaded GT: {len(df)} rows")
    return df


def match_predictions_to_gt(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
    time_threshold_ms: float = 2000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Сопоставление предсказаний с GT.

    Returns:
        (matched_pred, matched_gt, unmatched_pred)
    """
    matched_pred_indices = []
    matched_gt_indices = []

    for pred_idx, pred in predictions.iterrows():
        best_match = None
        best_score = 0

        # Ищем лучшее совпадение в GT
        for gt_idx, gt in ground_truth.iterrows():
            # Проверяем временной диапазон
            time_diff = abs(pred['frame_timestamp'] - gt['frame_timestamp'])
            if time_diff > time_threshold_ms:
                continue

            # Проверяем IoU
            pred_bbox = [pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']]
            gt_bbox = [gt['x_min'], gt['y_min'], gt['x_max'], gt['y_max']]

            iou = compute_iou(pred_bbox, gt_bbox)

            if iou > iou_threshold and iou > best_score:
                best_score = iou
                best_match = gt_idx

        if best_match is not None:
            matched_pred_indices.append(pred_idx)
            matched_gt_indices.append(best_match)

    # Создаем датафреймы
    matched_pred = predictions.loc[matched_pred_indices].reset_index(drop=True)
    matched_gt = ground_truth.loc[matched_gt_indices].reset_index(drop=True)

    unmatched_pred = predictions.drop(matched_pred_indices).reset_index(drop=True)
    unmatched_gt = ground_truth.drop(matched_gt_indices).reset_index(drop=True)

    return matched_pred, matched_gt, unmatched_pred


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Вычисление IoU для двух bbox."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def evaluate_field_accuracy(
    matched_pred: pd.DataFrame,
    matched_gt: pd.DataFrame,
    field: str
) -> float:
    """Оценка точности конкретного поля."""
    if field not in matched_pred.columns or field not in matched_gt.columns:
        return 0.0

    correct = 0
    total = len(matched_pred)

    for i in range(total):
        pred_val = str(matched_pred[field].iloc[i]).strip().lower()
        gt_val = str(matched_gt[field].iloc[i]).strip().lower()

        # Нормализация
        if field in ['price_default', 'price_card']:
            # Для цен сравниваем числовые значения
            try:
                pred_num = float(pred_val.replace(',', '.')) if pred_val else 0
                gt_num = float(gt_val.replace(',', '.')) if gt_val else 0
                # Допуск 1 рубль
                if abs(pred_num - gt_num) < 1:
                    correct += 1
            except:
                if pred_val == gt_val:
                    correct += 1
        else:
            # Точное совпадение для остальных полей
            if pred_val == gt_val and pred_val not in ['', 'nan', 'none', 'нет']:
                correct += 1

    return correct / total if total > 0 else 0


def evaluate_all(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_dir: Path = None
) -> Dict:
    """Полная оценка решения."""

    logger.info("=" * 60)
    logger.info("ОЦЕНКА ТОЧНОСТИ РЕШЕНИЯ")
    logger.info("=" * 60)

    # Загрузка данных
    predictions = load_predictions(predictions_csv)
    ground_truth = load_ground_truth(ground_truth_csv)

    # Сопоставление
    matched_pred, matched_gt, unmatched_pred = match_predictions_to_gt(
        predictions, ground_truth
    )

    logger.info(f"Matched: {len(matched_pred)} / GT: {len(ground_truth)}")
    logger.info(f"Unmatched predictions: {len(unmatched_pred)}")

    # Метрики детекции
    detection_recall = len(matched_gt) / len(ground_truth) if len(ground_truth) > 0 else 0
    detection_precision = len(matched_pred) / len(predictions) if len(predictions) > 0 else 0

    logger.info(f"\nDetection Recall: {detection_recall:.2%}")
    logger.info(f"Detection Precision: {detection_precision:.2%}")

    # Оценка полей
    fields_to_evaluate = [
        'barcode', 'product_name', 'price_default', 'price_card',
        'id_sku', 'discount_amount'
    ]

    field_accuracies = {}

    logger.info("\nТочность полей (только для matched):")
    for field in fields_to_evaluate:
        if field in matched_pred.columns:
            acc = evaluate_field_accuracy(matched_pred, matched_gt, field)
            field_accuracies[field] = acc
            logger.info(f"  {field}: {acc:.2%}")

    # Итоговая метрика (по заданию)
    # Ценник считается успешно распознанным если barcode или временная метка совпали
    # и >= 80% полей распознаны правильно

    successful_tags = 0
    for i in range(len(matched_pred)):
        fields_correct = 0
        for field in fields_to_evaluate:
            if field in matched_pred.columns:
                pred_val = str(matched_pred[field].iloc[i]).strip()
                gt_val = str(matched_gt[field].iloc[i]).strip()

                if field in ['price_default', 'price_card']:
                    try:
                        pred_num = float(pred_val.replace(',', '.')) if pred_val else 0
                        gt_num = float(gt_val.replace(',', '.')) if gt_val else 0
                        if abs(pred_num - gt_num) < 1:
                            fields_correct += 1
                    except:
                        pass
                else:
                    if pred_val.lower() == gt_val.lower() and pred_val not in ['', 'нет']:
                        fields_correct += 1

        accuracy = fields_correct / len(fields_to_evaluate)
        if accuracy >= 0.8:
            successful_tags += 1

    final_metric = successful_tags / len(ground_truth) if len(ground_truth) > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info(f"ИТОГОВАЯ МЕТРИКА: {final_metric:.2%}")
    logger.info(f"Успешно распознано: {successful_tags} / {len(ground_truth)}")
    logger.info("=" * 60)

    results = {
        'detection_recall': detection_recall,
        'detection_precision': detection_precision,
        'field_accuracies': field_accuracies,
        'final_metric': final_metric,
        'successful_tags': successful_tags,
        'total_gt_tags': len(ground_truth),
        'total_pred_tags': len(predictions),
        'matched_tags': len(matched_pred)
    }

    # Сохранение результатов
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Сохранение несовпадений для анализа
        unmatched_pred.to_csv(output_dir / 'unmatched_predictions.csv', index=False)
        matched_pred.to_csv(output_dir / 'matched_predictions.csv', index=False)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate price tag recognition')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='Path to ground truth CSV')
    parser.add_argument('--output', type=str, default='evaluation_output',
                        help='Output directory for results')

    args = parser.parse_args()

    results = evaluate_all(
        predictions_csv=Path(args.predictions),
        ground_truth_csv=Path(args.ground_truth),
        output_dir=Path(args.output)
    )

    print("\nОценка завершена!")
    print(f"Итоговая метрика: {results['final_metric']:.2%}")


if __name__ == '__main__':
    main()
