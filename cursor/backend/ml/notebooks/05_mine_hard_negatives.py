"""
Mine hard negatives and build YOLO train/val splits.

This script:
1) runs the current detector on labeled training frames,
2) extracts false-positive crops as empty-label negatives,
3) writes train/val txt files and a dedicated data_hnm.yaml config.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2


def _read_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            _, xc, yc, bw, bh = parts[:5]
            xc = float(xc) * img_w
            yc = float(yc) * img_h
            bw = float(bw) * img_w
            bh = float(bh) * img_h
        except Exception:
            continue

        x1 = max(0.0, xc - bw / 2.0)
        y1 = max(0.0, yc - bh / 2.0)
        x2 = min(float(img_w - 1), xc + bw / 2.0)
        y2 = min(float(img_h - 1), yc + bh / 2.0)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _expand_box(box: Tuple[float, float, float, float], pad_ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    nx1 = max(0, int(x1 - px))
    ny1 = max(0, int(y1 - py))
    nx2 = min(w, int(x2 + px))
    ny2 = min(h, int(y2 + py))
    return nx1, ny1, nx2, ny2


def mine_hard_negatives(
    dataset_dir: Path,
    model_path: Path,
    conf_thres: float = 0.35,
    fp_iou_thres: float = 0.15,
    max_new_negatives: int = 400,
    device: str = "cpu",
) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from exc

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
    if not images:
        return 0

    model = YOLO(str(model_path))
    created = 0

    # Avoid duplicate names from previous runs.
    existing_neg_idx = len(list(images_dir.glob("fpneg_*.jpg")))

    for img_path in images:
        if created >= max_new_negatives:
            break
        if img_path.stem.startswith("fpneg_"):
            continue

        frame = cv2.imread(str(img_path))
        if frame is None or frame.size == 0:
            continue
        h, w = frame.shape[:2]
        gt_boxes = _read_yolo_boxes(labels_dir / f"{img_path.stem}.txt", w, h)

        result = model.predict(source=frame, conf=conf_thres, iou=0.45, verbose=False, device=device)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        preds_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
        for pred in preds_xyxy:
            if created >= max_new_negatives:
                break
            px1, py1, px2, py2 = [float(v) for v in pred[:4]]
            if px2 <= px1 or py2 <= py1:
                continue
            pred_box = (px1, py1, px2, py2)

            best_iou = 0.0
            for gt_box in gt_boxes:
                best_iou = max(best_iou, _iou(pred_box, gt_box))
            if best_iou >= fp_iou_thres:
                continue

            x1, y1, x2, y2 = _expand_box(pred_box, pad_ratio=0.20, w=w, h=h)
            if x2 - x1 < 48 or y2 - y1 < 32:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Safety: reject crop if it intersects GT strongly.
            crop_box = (float(x1), float(y1), float(x2), float(y2))
            if any(_iou(crop_box, gt) > 0.05 for gt in gt_boxes):
                continue

            neg_name = f"fpneg_{existing_neg_idx + created:06d}"
            neg_img = images_dir / f"{neg_name}.jpg"
            neg_lbl = labels_dir / f"{neg_name}.txt"
            cv2.imwrite(str(neg_img), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            neg_lbl.write_text("", encoding="utf-8")
            created += 1

    return created


def build_train_val_files(
    dataset_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_negative_to_positive_ratio: float = 1.2,
) -> Tuple[int, int]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    positives: List[str] = []
    negatives: List[str] = []

    for img_path in sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")]):
        label_path = labels_dir / f"{img_path.stem}.txt"
        label_text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        rel_img = (dataset_dir / "images" / img_path.name).resolve().as_posix()
        if label_text:
            positives.append(rel_img)
        else:
            negatives.append(rel_img)

    rnd = random.Random(seed)
    rnd.shuffle(positives)
    rnd.shuffle(negatives)

    # Keep negatives bounded to avoid drowning positive samples.
    max_negatives = int(len(positives) * max_negative_to_positive_ratio)
    if max_negatives > 0:
        negatives = negatives[:max_negatives]

    val_pos = max(1, int(len(positives) * val_ratio)) if positives else 0
    val_neg = int(len(negatives) * val_ratio) if negatives else 0

    val_items = positives[:val_pos] + negatives[:val_neg]
    train_items = positives[val_pos:] + negatives[val_neg:]
    rnd.shuffle(train_items)
    rnd.shuffle(val_items)

    (dataset_dir / "train.txt").write_text("\n".join(train_items) + "\n", encoding="utf-8")
    (dataset_dir / "val.txt").write_text("\n".join(val_items) + "\n", encoding="utf-8")

    data_yaml = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: train.txt\n"
        f"val: val.txt\n\n"
        f"nc: 1\n"
        f"names: ['price_tag']\n"
    )
    (dataset_dir / "data_hnm.yaml").write_text(data_yaml, encoding="utf-8")
    return len(train_items), len(val_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for YOLO detector training")
    parser.add_argument("--dataset", type=str, default="backend/ml/data/processed")
    parser.add_argument("--model", type=str, default="backend/ml/models/price_tag_detector.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--fp-iou", type=float, default=0.15)
    parser.add_argument("--max-neg", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--neg-ratio", type=float, default=1.2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    model_path = Path(args.model)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    created = mine_hard_negatives(
        dataset_dir=dataset_dir,
        model_path=model_path,
        conf_thres=args.conf,
        fp_iou_thres=args.fp_iou,
        max_new_negatives=args.max_neg,
        device=args.device,
    )
    train_count, val_count = build_train_val_files(
        dataset_dir=dataset_dir,
        val_ratio=args.val_ratio,
        max_negative_to_positive_ratio=args.neg_ratio,
    )

    print(f"hard_negatives_created={created}")
    print(f"train_images={train_count}")
    print(f"val_images={val_count}")
    print(f"data_yaml={(dataset_dir / 'data_hnm.yaml').as_posix()}")


if __name__ == "__main__":
    main()
