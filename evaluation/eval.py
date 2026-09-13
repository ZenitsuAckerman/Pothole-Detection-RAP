import json
import os
import shutil
from ultralytics import RTDETR

WEIGHTS_PATH = "training/runs/train/pothole_rtdetr/weights/best.pt"
DATA_YAML = "data/data.yaml"
OUTPUT_DIR = "evaluation"
FAILURE_CASES_DIR = os.path.join(OUTPUT_DIR, "failure_cases")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def run_evaluation(model, split="val", output_name="metrics.json"):
    print(f"\n--- Running evaluation on '{split}' split ---")
    metrics = model.val(data=DATA_YAML, split=split)
    
    report = {
        "split": split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to {out_path}")
    print(json.dumps(report, indent=2))
    
    return metrics

def mine_failure_cases(model, split="test"):
    print(f"\n--- Mining failure cases on '{split}' split ---")
    ensure_dir(FAILURE_CASES_DIR)
    
    # We run predict with save=True, save_txt=True, save_conf=True
    # to get raw predictions we can compare against labels
    results = model.predict(
        source=f"data/{split}/images",
        conf=0.1,  # Low confidence to catch false positives and borderline cases
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_DIR,
        name="predict_test"
    )
    
    print("\nInference complete.")
    print(f"To mine failure cases, compare the predicted labels in {OUTPUT_DIR}/predict_test/labels/")
    print(f"with the ground truth labels in data/{split}/labels/.")
    print("Look for:")
    print("1. False positives: Prediction exists, but no ground truth.")
    print("2. False negatives: Ground truth exists, but no prediction.")
    print("3. Low confidence: Prediction confidence is very low (e.g. < 0.3).")
    print(f"\nCopy the 5 worst examples into {FAILURE_CASES_DIR}/ for your memo.")
    
def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}. Did you run training?")
        return
        
    model = RTDETR(WEIGHTS_PATH)
    
    # Run validation for tuning check
    run_evaluation(model, split="val", output_name="val_metrics.json")
    
    # Run test for honest reporting
    run_evaluation(model, split="test", output_name="test_metrics.json")
    
    # Mine failures on test set
    mine_failure_cases(model, split="test")

if __name__ == "__main__":
    main()
