#!/usr/bin/env python3
"""bbox2sam3 - YOLO Bbox to SAM3 Segmentation Masks

Converts YOLO bounding box annotations to segmentation masks using SAM3.
Supports three modes:
  - single: Standard SAM3 prediction (1 mask per bbox)
  - multi: 3 mask candidates, selects best by IOU score
  - iterative: Multiple refinement passes with mask feedback

Usage:
    python bbox2sam3.py --input ./examples/input --output ./examples/output \
        --checkpoint ~/projects/sam3/checkpoints/sam3.pt --classes 5

Author: UrbanVue
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> np.ndarray:
    """Converts YOLO normalized coordinates to XYXY pixel coordinates.

    Args:
        cx: Center x coordinate (normalized 0-1)
        cy: Center y coordinate (normalized 0-1)
        w: Width (normalized 0-1)
        h: Height (normalized 0-1)
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        numpy array [x0, y0, x1, y1] in pixels
    """
    x0 = (cx - w / 2) * img_w
    y0 = (cy - h / 2) * img_h
    x1 = (cx + w / 2) * img_w
    y1 = (cy + h / 2) * img_h
    return np.array([x0, y0, x1, y1])


def predict_single(model, inference_state, box: np.ndarray):
    """Standard SAM3 prediction with 1 output mask.

    Args:
        model: SAM3 image model
        inference_state: Output from processor.set_image()
        box: XYXY bbox as numpy array shape (4,)

    Returns:
        Tuple of (mask, score) or (None, None) on failure
    """
    masks, scores, _ = model.predict_inst(
        inference_state,
        box=box[None, :],
        multimask_output=False
    )

    if len(masks) == 0:
        return None, None

    mask = masks[0]
    if mask.ndim > 2:
        mask = mask.squeeze()

    return mask, scores[0]


def predict_multimask(model, inference_state, box: np.ndarray):
    """Multi-mask prediction: 3 candidates, selects best by IOU score.

    Args:
        model: SAM3 image model
        inference_state: Output from processor.set_image()
        box: XYXY bbox as numpy array shape (4,)

    Returns:
        Tuple of (best_mask, best_score, chosen_index) or (None, None, None)
    """
    masks, scores, _ = model.predict_inst(
        inference_state,
        box=box[None, :],
        multimask_output=True
    )

    if len(masks) == 0:
        return None, None, None

    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    if mask.ndim > 2:
        mask = mask.squeeze()

    return mask, scores[best_idx], best_idx


def predict_iterative(model, inference_state, box: np.ndarray, iterations: int = 2):
    """Iterative mask refinement with mask_input feedback.

    Args:
        model: SAM3 image model
        inference_state: Output from processor.set_image()
        box: XYXY bbox as numpy array shape (4,)
        iterations: Number of refinement passes (default 2)

    Returns:
        Tuple of (best_mask, best_score) or (None, None)

    Note:
        mask_input expects shape (1, 256, 256) - low-res logits
    """
    # Iteration 1: initial prediction
    masks, scores, low_res = model.predict_inst(
        inference_state,
        box=box[None, :],
        multimask_output=True
    )

    if len(masks) == 0:
        return None, None

    best_idx = np.argmax(scores)
    best_logits = low_res[best_idx:best_idx + 1]  # Shape (1, 256, 256)

    # Iterations 2..N: refine with previous low-res mask
    for _ in range(iterations - 1):
        masks, scores, low_res = model.predict_inst(
            inference_state,
            box=box[None, :],
            mask_input=best_logits,
            multimask_output=True
        )

        if len(masks) == 0:
            break

        best_idx = np.argmax(scores)
        best_logits = low_res[best_idx:best_idx + 1]

    mask = masks[best_idx]
    if mask.ndim > 2:
        mask = mask.squeeze()

    return mask, scores[best_idx]


# Default mask color (BGR format for OpenCV)
MASK_COLOR = (0, 255, 255)  # Yellow - good visibility on asphalt


def create_overlay(image: np.ndarray, combined_mask: np.ndarray,
                   alpha: float = 0.5) -> np.ndarray:
    """Creates an overlay of the mask on the original image.

    Args:
        image: Original image (BGR, uint8)
        combined_mask: Combined mask with class IDs as pixel values
        alpha: Overlay transparency (0-1, default 0.5)

    Returns:
        Image with semi-transparent mask overlay
    """
    overlay = image.copy()

    # Binary mask (all non-zero pixels)
    mask_binary = combined_mask > 0

    # Apply color to masked area
    overlay[mask_binary] = MASK_COLOR

    # Blend overlay with original
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Draw contours for better visibility
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, MASK_COLOR, 2)

    return result


def mask_to_contours(binary_mask: np.ndarray) -> list:
    """Extracts contours from binary mask for COCO segmentation.

    Args:
        binary_mask: Binary mask (uint8, 0 or 1)

    Returns:
        List of contours suitable for COCO segmentation
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cnt for cnt in contours if len(cnt) >= 3]


def parse_labels(label_path: Path, target_classes: set = None) -> list:
    """Parses YOLO label file and filters by classes.

    Args:
        label_path: Path to .txt label file
        target_classes: Set of class IDs to filter (None = all)

    Returns:
        List of tuples (class_id, cx, cy, w, h)
    """
    labels = []
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            continue

        if target_classes is not None and cls_id not in target_classes:
            continue

        cx, cy, w, h = map(float, parts[1:5])
        labels.append((cls_id, cx, cy, w, h))

    return labels


def discover_classes(input_dir: Path, image_ext: str) -> set:
    """Discovers all class IDs in the labels.

    Args:
        input_dir: Input directory with images and labels
        image_ext: Image file extension

    Returns:
        Set of found class IDs
    """
    all_classes = set()
    for img_path in input_dir.glob(f"*.{image_ext}"):
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        for cls_id, *_ in parse_labels(label_path):
            all_classes.add(cls_id)
    return all_classes


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO bounding boxes to segmentation masks using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with class filter
    python bbox2sam3.py -i ./input -o ./output -c sam3.pt --classes 5

    # Multi-mask mode (3 candidates, best IOU)
    python bbox2sam3.py -i ./input -o ./output -c sam3.pt --mode multi --classes 5 6

    # Iterative refinement with 3 passes
    python bbox2sam3.py -i ./input -o ./output -c sam3.pt --mode iterative --iterations 3
        """
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input directory with images (.jpg) and labels (.txt)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for masks and annotations"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to SAM3 checkpoint (sam3.pt)"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="single",
        choices=["single", "multi", "iterative"],
        help="Segmentation mode (default: single)"
    )
    parser.add_argument(
        "--iterations", type=int, default=2,
        help="Number of iterations for iterative mode (default: 2)"
    )
    parser.add_argument(
        "--classes", nargs="*", type=int, default=None,
        help="Filter by class IDs (e.g. 5 6 7)"
    )
    parser.add_argument(
        "--image-ext", type=str, default="jpg",
        help="Image extension (default: jpg)"
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Skip overlay visualization (default: overlay is generated)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay transparency 0-1 (default: 0.5)"
    )
    return parser.parse_args()


def main():
    """Main function for bbox2sam3 pipeline."""
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM3 model
    logger.info("Loading SAM3 model...")
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        enable_inst_interactivity=True
    )
    processor = Sam3Processor(model)
    logger.info("Model loaded on GPU")

    # Discover classes
    all_classes = discover_classes(input_dir, args.image_ext)
    target_classes = set(args.classes) if args.classes else all_classes

    logger.info(f"Found classes: {sorted(all_classes)}")
    logger.info(f"Processing classes: {sorted(target_classes)}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == "iterative":
        logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Overlay: {'no' if args.no_overlay else f'yes (alpha={args.alpha})'}")

    # COCO output structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c, "name": f"class_{c}"} for c in sorted(target_classes)]
    }
    ann_id = 1

    # Statistics
    stats = {
        "total_images": 0,
        "total_detections": 0,
        "total_pixels": 0,
        "mask_selection": {0: 0, 1: 0, 2: 0}  # For multi-mask mode
    }

    # Collect images
    images = sorted(input_dir.glob(f"*.{args.image_ext}"))
    logger.info(f"Found: {len(images)} images")

    for img_idx, img_path in enumerate(tqdm(images, desc="Processing")):
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        # Load image
        image = Image.open(img_path)
        img_w, img_h = image.size

        # Set image for SAM3
        inference_state = processor.set_image(image)

        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        image_id = img_idx + 1
        stats["total_images"] += 1

        # Process labels
        labels = parse_labels(label_path, target_classes)
        for cls_id, cx, cy, w, h in labels:
            box = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)

            # Predict mask based on mode
            if args.mode == "single":
                mask, score = predict_single(model, inference_state, box)
            elif args.mode == "multi":
                mask, score, mask_idx = predict_multimask(model, inference_state, box)
                if mask_idx is not None:
                    stats["mask_selection"][mask_idx] += 1
            else:  # iterative
                mask, score = predict_iterative(
                    model, inference_state, box, iterations=args.iterations
                )

            if mask is None:
                continue

            stats["total_detections"] += 1

            # Convert to binary mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            stats["total_pixels"] += np.sum(binary_mask)

            # Add to combined mask (class ID as pixel value)
            combined_mask[binary_mask > 0] = cls_id % 256

            # COCO polygon annotations
            contours = mask_to_contours(binary_mask)
            for cnt in contours:
                segmentation = cnt.flatten().tolist()
                bx, by, bw, bh = cv2.boundingRect(cnt)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": [segmentation],
                    "bbox": [bx, by, bw, bh],
                    "area": float(np.sum(binary_mask)),
                    "iscrowd": 0
                })
                ann_id += 1

        # Save mask
        mask_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), combined_mask)

        # Create and save overlay (unless --no-overlay)
        if not args.no_overlay:
            # Load original image as BGR for OpenCV
            img_bgr = cv2.imread(str(img_path))
            overlay = create_overlay(img_bgr, combined_mask, alpha=args.alpha)
            overlay_path = output_dir / f"{img_path.stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)

        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h
        })

    # Save COCO JSON
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    # Report results
    logger.info("\nDone!")
    logger.info(f"  Masks: {output_dir}/*_mask.png")
    if not args.no_overlay:
        logger.info(f"  Overlays: {output_dir}/*_overlay.jpg")
    logger.info(f"  Annotations: {annotations_path}")
    logger.info(f"\nStatistics:")
    logger.info(f"  Processed images: {stats['total_images']}")
    logger.info(f"  Total detections: {stats['total_detections']}")
    logger.info(f"  Total mask pixels: {stats['total_pixels']:,}")

    if args.mode == "multi" and stats["total_detections"] > 0:
        logger.info("\nMulti-mask selection:")
        for idx, count in stats["mask_selection"].items():
            pct = count / stats["total_detections"] * 100
            logger.info(f"  Mask {idx}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
