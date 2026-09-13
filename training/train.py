import json
import time
import subprocess
import platform
import argparse
import os
from ultralytics import RTDETR

CONFIG = {
    "model": "rtdetr-l.pt",
    "data": "data/data.yaml",
    "epochs": 30,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "seed": 42,
    "patience": 15,
    "mosaic": 0.5,          # Moderate, not maximal
    "close_mosaic": 10,     # Disable mosaic for final 10 epochs
    "mixup": 0.0,           # Disabled - needs more epochs
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,           # Cheap, effective lighting robustness
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.3,           # Light spatial aug only
}

def remove_excluded_boxes(data_yaml_dir, excluded_json_path):
    """
    Reads the excluded_small_boxes.json and removes those specific lines 
    from the corresponding label files.
    This modifies the files in place, so it should only be done if requested.
    """
    if not os.path.exists(excluded_json_path):
        print(f"Warning: {excluded_json_path} not found. Skipping box exclusion.")
        return

    print(f"Loading exclusions from {excluded_json_path}...")
    with open(excluded_json_path, 'r') as f:
        excluded_boxes = json.load(f)
    
    # Group by file
    exclusions_by_file = {}
    for box in excluded_boxes:
        filepath = box['label_file']
        if filepath not in exclusions_by_file:
            exclusions_by_file[filepath] = set()
        exclusions_by_file[filepath].add(box['line_index'])

    removed_count = 0
    for filepath, indices_to_remove in exclusions_by_file.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            with open(filepath, 'w') as f:
                for i, line in enumerate(lines):
                    if i not in indices_to_remove:
                        f.write(line)
                    else:
                        removed_count += 1

    print(f"Removed {removed_count} small boxes from training data.")

def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR for pothole detection.")
    parser.add_argument("--exclude-small-boxes", action="store_true", help="Remove small boxes flagged by curate_dataset.py before training.")
    args = parser.parse_args()

    if args.exclude_small_boxes:
        excluded_json = os.path.join(os.path.dirname(CONFIG['data']), "excluded_small_boxes.json")
        print("Exclude small boxes flag is SET. Modifying label files...")
        remove_excluded_boxes(os.path.dirname(CONFIG['data']), excluded_json)
        print("Note: This modifies the label files directly.")
    else:
        print("Exclude small boxes flag is NOT set. Training on full dataset including small boxes.")

    print("\nStarting training...")
    start_time = time.time()
    
    model = RTDETR(CONFIG["model"])
    
    # Pass all config keys to the train function except 'model'
    train_args = {k: v for k, v in CONFIG.items() if k != 'model'}
    train_args['project'] = "training/runs/train"
    train_args['name'] = "pothole_rtdetr"
    
    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    elapsed = time.time() - start_time

    # Collect reproducibility metadata
    try:
        gpu_info = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").strip()
    except:
        gpu_info = "Unknown / CPU"
        
    try:
        ul_version = subprocess.getoutput("pip show ultralytics | grep Version").strip().split(": ")[1]
    except:
        ul_version = "Unknown"

    log = {
        **CONFIG,
        "excluded_small_boxes": args.exclude_small_boxes,
        "wall_clock_seconds": elapsed,
        "python": platform.python_version(),
        "gpu": gpu_info,
        "ultralytics_version": ul_version,
    }
    
    log_path = "training/hyperparams.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
        
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"Best weights at: training/runs/train/pothole_rtdetr/weights/best.pt")
    print(f"Hyperparameters saved to: {log_path}")

if __name__ == "__main__":
    main()
