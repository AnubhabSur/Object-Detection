# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
YOLO Image Inference Script
Runs a trained Ultralytics YOLO model on an image file and saves annotated output.
"""

import sys
from pathlib import Path


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

def get_inputs():
    while True:
        image = input("Enter input image path: ").strip()
        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            print("  [ERROR] Image not found: {}. Try again.".format(image_path))
            continue
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            print("  [ERROR] Unsupported format '{}'. Supported: {}".format(
                image_path.suffix, ", ".join(SUPPORTED_EXTS)))
            continue
        break

    while True:
        model = input("Enter model weights path (.pt): ").strip()
        model_path = Path(model).expanduser().resolve()
        if model_path.exists():
            break
        print("  [ERROR] Model not found: {}. Try again.".format(model_path))

    OUTPUT_DIR = Path(r"D:\Object Detection\output")
    output_path = OUTPUT_DIR / "{}_annotated{}".format(image_path.stem, image_path.suffix)

    conf_raw = input("Confidence threshold [0.25]: ").strip()
    try:
        conf = float(conf_raw) if conf_raw else 0.25
    except ValueError:
        print("  [WARN] Invalid value, defaulting to 0.25")
        conf = 0.25

    return image_path, model_path, output_path, conf


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

def resolve_device():
    try:
        import torch
        if torch.cuda.is_available():
            print("  [OK] CUDA detected - using GPU 0")
            return "0"
    except ImportError:
        pass

    print("  [WARN] No CUDA detected - using CPU")
    return "cpu"


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------

def run_inference(image_path, model_path, output_path, conf, device):
    try:
        import cv2
    except ImportError:
        print("  [ERROR] opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load model
    print("\n  Loading model: {}".format(model_path))
    model = YOLO(str(model_path))
    print("  [OK] Model loaded")

    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("  [ERROR] Could not read image: {}".format(image_path))
        sys.exit(1)

    height, width = frame.shape[:2]
    print("  [OK] Image loaded: {}x{}".format(width, height))
    print("  Output -> {}\n".format(output_path))

    # Run inference
    results = model(frame, conf=conf, device=device, verbose=False)
    annotated = results[0].plot()

    # Save output
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        print("  [ERROR] Could not save output image: {}".format(output_path))
        sys.exit(1)

    # Print detection summary
    boxes = results[0].boxes
    n_detections = len(boxes) if boxes is not None else 0
    print("  [OK] Detections found: {}".format(n_detections))
    print("  [OK] Done! Annotated image saved to: {}".format(output_path))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\n+======================================+")
    print("|      YOLO Image Inference Script     |")
    print("+======================================+\n")

    image_path, model_path, output_path, conf = get_inputs()
    device = resolve_device()

    print("\n-- Summary --------------------------------")
    print("  Image  : {}".format(image_path))
    print("  Model  : {}".format(model_path))
    print("  Output : {}".format(output_path))
    print("  Conf   : {}".format(conf))
    print("  Device : {}".format(device))

    go = input("\n>> Start inference? [Y/n]: ").strip().lower()
    if go == "n":
        print("  Aborted.")
        return

    run_inference(image_path, model_path, output_path, conf, device)


if __name__ == "__main__":
    main()