# bbox2sam3

Convert YOLO bounding box annotations to segmentation masks using Meta's SAM3. Useful when you have object detections but need pixel-level masks for training or visualization.

## What it does

Given a folder of images with YOLO-format label files, this tool:
1. Loads each bounding box annotation
2. Runs SAM3 to generate a segmentation mask for that region
3. Outputs combined mask images and COCO-format polygon annotations
4. Optionally creates overlay visualizations

## Installation

You'll need Python 3.10+, PyTorch with CUDA, and SAM3 installed locally.

```bash
# Assumes SAM3 is at ~/projects/sam3
source ~/dev/venv/bin/activate
pip install -r requirements.txt
```

## Basic usage

```bash
python bbox2sam3.py \
    --input examples/input \
    --output examples/output \
    --checkpoint ~/projects/sam3/checkpoints/sam3.pt \
    --classes 5
```

The `--classes` flag filters which class IDs to process. Omit it to process all detections.

## Prediction modes

The tool supports three different ways to generate masks:

**Single mode** (default) — Fastest option. SAM3 produces one mask per bounding box. Good enough for most objects with clear boundaries.

**Multi mode** — SAM3 generates three mask candidates per bbox and picks the one with the best IOU score. Takes longer but handles ambiguous boundaries better.

```bash
python bbox2sam3.py --mode multi ...
```

**Iterative mode** — Runs multiple refinement passes, feeding the previous mask back into SAM3. Best for thin objects like cracks where the initial mask might miss parts.

```bash
python bbox2sam3.py --mode iterative --iterations 3 ...
```

## Input format

Put your images and YOLO label files in the same folder:

```
input/
├── frame_0001.jpg
├── frame_0001.txt
├── frame_0002.jpg
└── frame_0002.txt
```

Each `.txt` file contains YOLO-format annotations (class_id, center_x, center_y, width, height — all normalized 0-1):

```
5 0.5 0.3 0.1 0.2
6 0.7 0.6 0.15 0.1
```

## Output format

```
output/
├── frame_0001_mask.png      # Combined mask (pixel value = class ID)
├── frame_0001_overlay.jpg   # Original image with mask overlay
├── frame_0002_mask.png
├── frame_0002_overlay.jpg
└── annotations.json         # COCO format polygons
```

The mask PNGs are 8-bit grayscale where each pixel's value represents the class ID (mod 256). Background is 0. The `annotations.json` file contains COCO-format segmentation polygons that you can use directly for training.

## All options

```
--input, -i       Input directory with images and labels (required)
--output, -o      Output directory for masks (required)
--checkpoint, -c  Path to SAM3 checkpoint (required)
--mode, -m        single, multi, or iterative (default: single)
--iterations      Refinement passes for iterative mode (default: 2)
--classes         Filter by class IDs (space-separated)
--image-ext       Image file extension (default: jpg)
--no-overlay      Skip generating overlay visualizations
--alpha           Overlay transparency 0-1 (default: 0.5)
```

## License

Internal use — UrbanVue
