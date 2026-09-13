# Pothole Detection & Reasoning API

A production-ready FastAPI service combining RT-DETR object detection with a Groq-powered reasoning layer.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Copy `.env.example` to `.env` and fill in your API keys:
   - `ROBOFLOW_API_KEY`: Needed to download the dataset.
   - `GROQ_API_KEY`: Needed for the reasoning API (Part B).

## Workflow

> **Architecture Note:** Dataset labels are stored in YOLO-format annotation files (`data.yaml` + `.txt` boxes) because that is the export convention Roboflow and Ultralytics share. The model trained is **RT-DETR** (`ultralytics.RTDETR("rtdetr-l.pt")`, a transformer-based detector), not a YOLO architecture.

### 1. Download & Curate Dataset
```bash
python scripts/fetch_dataset.py
python data/curate_dataset.py
```
This downloads a verified pothole dataset from Roboflow and flags small bounding boxes (<0.1% area) for exclusion if desired.

### 2. Train RT-DETR
**(GPU Required)**
```bash
python training/train.py
```
This trains the model using a moderate augmentation config suitable for a short 30-epoch run.

### 3. Evaluate & Mine Failure Cases
```bash
python evaluation/eval.py
```
This evaluates the model on the `test` split to generate honest metrics for the memo, and automatically mines failure cases (false positives/negatives, low confidence) into `evaluation/failure_cases/`.

### 4. Upload Weights (For Reviewers)
Upload your `best.pt` to Hugging Face Hub:
```bash
huggingface-cli login
python scripts/upload_weights.py --repo-id <your-username>/pothole-rtdetr
```

## Running the API

Start the FastAPI server:
```bash
python -m api.main
```

### Endpoints

#### `POST /detect`
Upload an image to get structured detection output.
```bash
curl -X POST "http://localhost:8000/detect" -F "file=@data/test/images/sample.jpg"
```

#### `POST /ask`
Ask a natural language question about the image.
```bash
curl -X POST "http://localhost:8000/ask?question=How%20many%20potholes%20are%20there?" -F "file=@data/test/images/sample.jpg"
```

## Reviewer: Downloading Weights
To run this API with the pre-trained weights, execute this Python snippet:
```python
from huggingface_hub import hf_hub_download
import os

os.makedirs("weights", exist_ok=True)
hf_hub_download(repo_id="<your-username>/pothole-rtdetr", filename="best.pt", local_dir="weights")
```
*(Replace `<your-username>` with the actual repo ID provided in the submission).*
