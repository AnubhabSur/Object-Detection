# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
YOLO Inference Script
Runs a trained Ultralytics YOLO model on an image or video file and saves annotated output.
"""

import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
OUTPUT_DIR = Path(r"D:\Object Detection\output")


# ---------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------

def select_mode():
    while True:
        mode = input("Inference mode - [1] Image  [2] Video : ").strip()
        if mode == "1":
            print("  [OK] Image inference selected\n")
            return "image"
        elif mode == "2":
            print("  [OK] Video inference selected\n")
            return "video"
        print("  [ERROR] Enter 1 for Image or 2 for Video.")


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

def get_inputs(mode):
    # Input file
    supported = IMAGE_EXTS if mode == "image" else VIDEO_EXTS
    label = "image" if mode == "image" else "video"

    while True:
        raw = input("Enter input {} path: ".format(label)).strip()
        input_path = Path(raw).expanduser().resolve()
        if not input_path.exists():
            print("  [ERROR] File not found: {}. Try again.".format(input_path))
            continue
        if input_path.suffix.lower() not in supported:
            print("  [ERROR] Unsupported format '{}'. Supported: {}".format(
                input_path.suffix, ", ".join(sorted(supported))))
            continue
        break

    # Model weights
    while True:
        model = input("Enter model weights path (.pt): ").strip()
        model_path = Path(model).expanduser().resolve()
        if model_path.exists():
            break
        print("  [ERROR] Model not found: {}. Try again.".format(model_path))

    # Output path (fixed directory, annotated suffix)
    out_ext = input_path.suffix if mode == "image" else ".mp4"
    output_path = OUTPUT_DIR / "{}_annotated{}".format(input_path.stem, out_ext)

    # Confidence threshold
    conf_raw = input("Confidence threshold [0.25]: ").strip()
    try:
        conf = float(conf_raw) if conf_raw else 0.25
    except ValueError:
        print("  [WARN] Invalid value, defaulting to 0.25")
        conf = 0.25

    return input_path, model_path, output_path, conf


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
# IMAGE INFERENCE
# ---------------------------------------------------------

def run_image_inference(input_path, model_path, output_path, conf, device):
    import cv2
    from ultralytics import YOLO

    print("\n  Loading model: {}".format(model_path))
    model = YOLO(str(model_path))
    print("  [OK] Model loaded")

    frame = cv2.imread(str(input_path))
    if frame is None:
        print("  [ERROR] Could not read image: {}".format(input_path))
        sys.exit(1)

    height, width = frame.shape[:2]
    print("  [OK] Image loaded: {}x{}".format(width, height))
    print("  Output -> {}\n".format(output_path))

    results = model(frame, conf=conf, device=device, verbose=False)
    annotated = results[0].plot()

    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        print("  [ERROR] Could not save output image: {}".format(output_path))
        sys.exit(1)

    boxes = results[0].boxes
    n_detections = len(boxes) if boxes is not None else 0
    print("  [OK] Detections found: {}".format(n_detections))
    print("  [OK] Done! Annotated image saved to: {}".format(output_path))


# ---------------------------------------------------------
# VIDEO INFERENCE
# ---------------------------------------------------------

def run_video_inference(input_path, model_path, output_path, conf, device):
    import cv2
    from ultralytics import YOLO

    print("\n  Loading model: {}".format(model_path))
    model = YOLO(str(model_path))
    print("  [OK] Model loaded")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("  [ERROR] Could not open video: {}".format(input_path))
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  [OK] Video opened: {}x{}  {:.1f} fps  {} frames".format(width, height, fps, total))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("  [ERROR] Could not open output writer: {}".format(output_path))
        sys.exit(1)

    print("  Output -> {}\n".format(output_path))

    frame_idx = 0
    log_every = max(1, total // 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, device=device, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % log_every == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total > 0 else 0
            print("  [{:5d}/{:5d}]  {:.1f}%".format(frame_idx, total, pct))

    cap.release()
    writer.release()
    print("\n  [OK] Done! Annotated video saved to: {}".format(output_path))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\n+======================================+")
    print("|       YOLO Inference Script          |")
    print("+======================================+\n")

    # Check dependencies upfront
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

    mode = select_mode()
    input_path, model_path, output_path, conf = get_inputs(mode)
    device = resolve_device()

    print("\n-- Summary --------------------------------")
    print("  Mode   : {}".format(mode.capitalize()))
    print("  Input  : {}".format(input_path))
    print("  Model  : {}".format(model_path))
    print("  Output : {}".format(output_path))
    print("  Conf   : {}".format(conf))
    print("  Device : {}".format(device))

    go = input("\n>> Start inference? [Y/n]: ").strip().lower()
    if go == "n":
        print("  Aborted.")
        return

    if mode == "image":
        run_image_inference(input_path, model_path, output_path, conf, device)
    else:
        run_video_inference(input_path, model_path, output_path, conf, device)


if __name__ == "__main__":
    main()