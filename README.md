<div align="center">

# 🕳️ Pothole Detection & Reasoning API

### RT-DETR-L Computer Vision + Groq-Powered Visual Reasoning

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![RT-DETR](https://img.shields.io/badge/Vision-RT--DETR--L-FF6F00?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Reasoning-Groq-F55036?style=for-the-badge)](https://groq.com/)
[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE?style=for-the-badge)](https://roboflow.com/)

**Computer Vision + Applied ML Engineering Take-Home Project**

</div>

---

## 📌 Executive Summary

This project implements a domain-specific **pothole detection and visual reasoning system** using **RT-DETR-L**, fine-tuned on a pothole detection dataset and exposed through a **FastAPI** service.

The system has two core capabilities:

1. **Object Detection** — Detect potholes and return structured bounding boxes and confidence scores.
2. **Natural-Language Reasoning** — Answer questions about the image using a lightweight hand-written intent router and a Groq-powered reasoning layer operating over structured detector output.

The system is intentionally implemented without agentic frameworks such as LangChain, LangGraph, CrewAI, or AutoGen.

### Core Pipeline

```text
                         INPUT IMAGE
                              │
                              ▼
                    ┌──────────────────┐
                    │     RT-DETR-L    │
                    │  Pothole Detector│
                    └────────┬─────────┘
                             │
                             ▼
                 Structured Detection Data
                 ┌────────────────────────┐
                 │ • Class                │
                 │ • Bounding Box         │
                 │ • Confidence            │
                 │ • Object Count          │
                 └───────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Intent Router  │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
             Detection Query      Reasoning Query
                   │                   │
                   │            ┌──────▼──────┐
                   │            │  Groq LLM   │
                   │            │  Reasoning  │
                   │            └──────┬──────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
                       API Response
```

## 🎯 Key Features
- **RT-DETR-L** fine-tuned for pothole detection
- **Domain-specific non-COCO detection class:** `pothole`
- **Roboflow-sourced dataset**
- **FastAPI inference service**
- **Structured detection responses**
- Bounding boxes and confidence scores
- **Natural-language image questions**
- **Hand-written deterministic intent routing**
- **Groq-powered reasoning over structured CV evidence** (`openai/gpt-oss-120b`)
- **Confidence-based reasoning guardrail**
- Explicit insufficient information behavior
- Automatic failure-case mining
- Reproducible training configuration
- No LangChain / LangGraph / CrewAI / AutoGen
- Model weights can be distributed separately from source code

> **Architecture Note:** The dataset uses YOLO-format annotation files (`data.yaml` + `.txt` bounding-box labels) because this is the export convention used by Roboflow and consumed by the Ultralytics training pipeline. The actual detector is **RT-DETR-L**, not a YOLO architecture.

## ✅ Problem Statement Compliance
| Requirement | Implementation |
|---|---|
| RT-DETR object detector | RT-DETR-L fine-tuned using Ultralytics |
| Non-COCO class | `pothole` |
| Custom domain dataset | Roboflow pothole detection dataset |
| Train / Validation / Test | Dataset split provided by the dataset |
| Object detection API | `POST /detect` |
| Natural-language reasoning | `POST /ask` |
| Intent routing | Hand-written decision layer |
| Structured reasoning | Reasoning operates over detector output |
| Confidence guardrail | Explicit insufficient-information behavior |
| Agentic frameworks | None |
| AutoML / No-Code | None |
| Reproducibility | Fixed seed and documented training configuration |
| Failure analysis | Automated failure-case mining + manual analysis |

## 📊 Evaluation Results

| Metric | Result |
|---|---|
| mAP@50 | **0.829** |
| mAP@50-95 | **0.494** |
| Precision | **0.817** |
| Recall | **0.713** |
| Test Images | 67 |
| Training Duration | ~25.8 minutes |

The evaluation pipeline reports detection performance on the held-out test split and is also used to identify representative failure cases.

## 🔬 Dataset

The project uses a pothole detection dataset sourced from Roboflow Universe (`yolo-sfvlm/pothole-detection-using-yolov5-p20qq`).

**Dataset Characteristics**
- **Domain:** Road / pothole detection
- **Target class:** `pothole`
- **Annotation format:** YOLO
- Images are divided into training, validation, and test splits (465 train, 133 valid, 67 test)
- Bounding-box annotations are used for object detection

Dataset acquisition is reproducible through:

```bash
python scripts/fetch_dataset.py
```

The dataset source and version are documented in the project configuration and memo.

## 🧠 Model

The detector uses **RT-DETR-L** (Real-Time Detection Transformer - Large).

```python
from ultralytics import RTDETR
model = RTDETR("rtdetr-l.pt")
```

RT-DETR was selected because the task requires an object detector capable of handling varying pothole sizes and visual conditions while providing direct bounding-box predictions.

**Training Configuration**
| Parameter | Configuration |
|---|---|
| Model | RT-DETR-L |
| Image Size | 640 × 640 |
| Epochs | 30 |
| Batch Size | 8 |
| Optimizer | AdamW / Ultralytics configuration |
| Seed | 42 |
| Deterministic | Yes |
| Hardware | Tesla T4 (Kaggle) |

Training is performed using:

```bash
python training/train.py
```

The best checkpoint is saved under the training run's `weights/best.pt`.

## 🔍 Evaluation & Failure Analysis

Evaluation is performed using:

```bash
python evaluation/eval.py
```

The evaluation pipeline measures:
- mAP@50
- mAP@50-95
- Precision
- Recall
- Confusion behavior
- Detection confidence

It also mines representative failure cases including:
- False positives
- False negatives
- Low-confidence detections

These cases are used for qualitative root-cause analysis rather than hiding model weaknesses. See `MEMO.md` for the analysis of the 5 specific failure cases identified.

## 🤖 Part B — Natural Language Reasoning

The `/ask` endpoint allows users to ask natural-language questions about an uploaded image.

Example:
**"How many potholes are there?"**

The request passes through a simple hand-written intent router.

```text
User Question
      │
      ▼
Intent Router
      │
      ├── Detection information already sufficient
      │          │
      │          ▼
      │     Structured Answer
      │
      └── Visual evidence required
                 │
                 ▼
            RT-DETR Detection
                 │
                 ▼
        Structured Detection Data
                 │
                 ▼
             Groq Reasoning
                 │
                 ▼
             Final Answer
```

The reasoning model does not directly perform object detection. Instead, it receives structured evidence such as:

```json
{
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.91,
      "bbox": [120, 180, 340, 420]
    }
  ],
  "count": 1
}
```

This keeps the computer-vision decision separate from the language reasoning layer.

### 🛡️ Confidence Guardrail

The reasoning layer includes a confidence-based guardrail. If the detector does not provide sufficient evidence to confidently answer a visual question, the system does not invent an answer. Instead, it explicitly reports that the available visual evidence is insufficient.

**Example:**
Question: *"How deep is this pothole?"*

If detector evidence is insufficient:
Answer: *"Insufficient information: this model only detects pothole presence and location, not attributes like depth, age, or cause."*

This prevents the LLM from hallucinating visual information that was not supported by the detector.

## 🚀 Quickstart

**1. Clone the Repository**
```bash
git clone https://github.com/ZenitsuAckerman/Pothole-Detection-RAP.git
cd Pothole-Detection-RAP
```

**2. Create a Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```
Configure your `ROBOFLOW_API_KEY` and `GROQ_API_KEY`. Never commit `.env` or API keys to the repository.

**5. Download Dataset**
```bash
python scripts/fetch_dataset.py
```

**6. Run FastAPI**
Start the FastAPI application:
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
The API will be available at: http://localhost:8000
Interactive Swagger documentation: http://localhost:8000/docs

## 📡 API

### `POST /detect`
Upload an image and receive structured pothole detections.

**Request**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_img.jpg"
```
**Example Response**
```json
{
  "detections": [
    {
      "class_name": "pothole",
      "confidence": 0.91,
      "bbox": [120, 180, 340, 420]
    }
  ]
}
```

### `POST /ask`
Ask a natural-language question about the uploaded image.

**Request**
```bash
curl -X POST \
  "http://localhost:8000/ask?question=How%20many%20potholes%20are%20there%3F" \
  -F "file=@test_img.jpg"
```
**Example Response**
```json
{
  "answer": "The image contains 1 detected pothole.",
  "detections": [
    {
      "class_name": "pothole",
      "confidence": 0.91,
      "bbox": [120, 180, 340, 420]
    }
  ]
}
```

## 🔁 Reproducibility

The project is designed so that another developer can reproduce the complete pipeline:

```text
Install Dependencies
        │
        ▼
Download Dataset
        │
        ▼
Train RT-DETR
        │
        ▼
Evaluate Test Split
        │
        ▼
Analyze Failure Cases
        │
        ▼
Load best.pt
        │
        ▼
Run FastAPI
```

Key reproducibility settings include:
- Fixed seed: 42
- Deterministic training
- Explicit image size
- Explicit epoch count
- Documented dataset source/version
- Defined training script
- Defined evaluation script
- Environment variable based API configuration

## 📦 Model Weights

The trained `best.pt` checkpoint is distributed separately from the source repository when required due to GitHub file-size constraints. Once uploaded, reviewers can download the checkpoint using:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="<your-username>/pothole-rtdetr",
    filename="best.pt",
    local_dir="weights"
)
```
Then configure the API to load `weights/best.pt`.

## ⚙️ Engineering Decisions

**Why RT-DETR?**
RT-DETR provides a transformer-based object detection architecture suitable for real-time detection while avoiding reliance on a YOLO detector.

**Why a separate reasoning layer?**
The LLM is intentionally separated from visual detection. RT-DETR answers: "What objects were detected and where?" The reasoning layer answers: "What does this structured evidence imply for the user's question?" This separation makes the system easier to inspect, test, and reason about.

**Why no agentic framework?**
The problem requires a lightweight decision layer rather than multi-agent orchestration. Therefore the system uses a Hand-Written Intent Router -> RT-DETR -> Structured Evidence -> Groq Reasoning with no LangChain, LangGraph, CrewAI, AutoGen, or similar agentic framework.

## 🔐 Security & Reliability
- API keys are stored through environment variables.
- Secrets are excluded from version control.
- Uploaded images are processed through the API inference pipeline.
- LLM reasoning is grounded in structured detector output.
- Low-confidence evidence triggers the insufficient-information guardrail.
- The detector and reasoning layers remain independently testable.

## 📌 Current Project Status

| Component | Status |
|---|---|
| Dataset sourcing | ✅ Complete |
| Dataset preparation | ✅ Complete |
| RT-DETR training pipeline | ✅ Complete |
| RT-DETR training | ✅ Complete |
| Test evaluation | ✅ Complete |
| Failure-case analysis | ✅ Complete |
| `/detect` API | ✅ Implemented |
| `/ask` reasoning API | ✅ Implemented |
| Confidence guardrail | ✅ Implemented |
| Model weight publishing | ⏳ Pending final checkpoint |

## 📄 License
This project is intended as an individual technical take-home submission.
