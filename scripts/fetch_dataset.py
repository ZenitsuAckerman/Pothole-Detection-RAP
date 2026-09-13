import argparse
import os
import sys

try:
    from roboflow import Roboflow
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Error: 'roboflow' or 'python-dotenv' library not found.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow Universe.")
    parser.add_argument("--workspace", type=str, default="yolo-sfvlm", help="Roboflow workspace slug")
    parser.add_argument("--project", type=str, default="pothole-detection-using-yolov5-p20qq", help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=2, help="Dataset version number")
    parser.add_argument("--format", type=str, default="yolov8", help="Download format (e.g., yolov8)")
    parser.add_argument("--dest", type=str, default="data", help="Destination folder (default: data)")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable is not set.")
        sys.exit(1)

    print(f"Connecting to Roboflow... (Workspace: {args.workspace}, Project: {args.project}, Version: {args.version})")
    
    rf = Roboflow(api_key=api_key)
    
    try:
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download(args.format, location=args.dest)
        
        print("\n--- DATASET DETAILS ---")
        print(f"Location: {dataset.location}")
        # Note: We rely on the user to visually confirm the license on the project page.
        # This script downloads it and sets it up for training.
        print("\nDownload complete. Make sure to run data/curate_dataset.py next to flag small bounding boxes if desired.")
        
    except Exception as e:
        print(f"\nFailed to download dataset. Error: {e}")
        print("\nDid you confirm the version number is correct on the project page?")

if __name__ == "__main__":
    main()
