import os
import argparse
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: 'huggingface_hub' library not found. Install it with: pip install huggingface_hub")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Upload trained weights to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Hub repo ID, e.g. username/pothole-rtdetr")
    parser.add_argument("--weights-path", type=str, default="training/runs/train/pothole_rtdetr/weights/best.pt", help="Path to best.pt")
    args = parser.parse_args()

    if not os.path.exists(args.weights_path):
        print(f"Error: Weights not found at {args.weights_path}.")
        sys.exit(1)

    print(f"Uploading {args.weights_path} to HF Hub repo '{args.repo_id}'...")
    
    try:
        api = HfApi()
        # Note: requires the user to be logged in via `huggingface-cli login` or have HF_TOKEN set
        url = api.upload_file(
            path_or_fileobj=args.weights_path,
            path_in_repo="best.pt",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("\nUpload successful!")
        print(f"Direct URL: {url}")
        print("\nAdd the following to your README instructions for the reviewer:")
        print("```python")
        print("from huggingface_hub import hf_hub_download")
        print(f"path = hf_hub_download(repo_id='{args.repo_id}', filename='best.pt')")
        print("```")
    except Exception as e:
        print(f"\nUpload failed. Error: {e}")
        print("Did you authenticate with Hugging Face? Run: huggingface-cli login")

if __name__ == "__main__":
    main()
