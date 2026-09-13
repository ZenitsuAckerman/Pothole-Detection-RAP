# Pothole Detection & Reasoning API
### Technical Screening Memo

## 1. Domain & Dataset

**Domain:** Road Infrastructure Safety — Pothole Detection

**Why this domain:**  
Potholes are a practical road-safety and infrastructure-maintenance problem where
localized object detection can support automated road inspection.

**Dataset source:**  
Roboflow Universe — yolo-sfvlm/pothole-detection-using-yolov5-p20qq version 2

**Dataset composition:**  
- Images: 665
- Class: `pothole`
- Annotation format: YOLO bounding boxes
- Non-COCO class requirement: satisfied because `pothole` is not a standard COCO class.

**Dataset preparation:**  
We maintained the original dataset images and labels without removing any boxes. We initially considered filtering bounding boxes below 0.1% area but decided against it to maintain the integrity of the test set and evaluate true model limits.

---

## 2. Split Strategy & Training

**Split:**  
- Train: 465 images
- Validation: 133 images
- Test: 67 images

**Justification:**  
The training split was used for model fitting, validation for model selection/
early stopping, and the test split was kept for final evaluation.

**Model:** RT-DETR-L (`ultralytics.RTDETR`)

**Training configuration:**

| Parameter | Actual value |
|---|---|
| Epochs | 30 |
| Image size | 640 |
| Batch size | 8 |
| Optimizer | `auto` → AdamW (lr 0.002, momentum 0.9) |
| Seed | 42 |
| Deterministic | True |
| Hardware | Tesla T4 |
| Training time | ~25.8 minutes |

---

## 3. Evaluation & Interpretation

### Results

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Validation | 0.774 | 0.453 | 0.817 | 0.646 |
| Test | 0.829 | 0.494 | 0.817 | 0.713 |

**Interpretation:**  
The model demonstrates strong pothole detection capabilities with a test mAP50 of 0.829, indicating it reliably finds most potholes with good confidence. The drop in mAP50-95 to 0.494 highlights that exact bounding box tightness varies, which is expected for irregularly shaped objects like potholes. The test metrics slightly outperform validation, suggesting the model generalized well without overfitting.

**Limitations:**  
These metrics measure performance on the available evaluation data and do not
guarantee performance on the private hidden evaluation set. In particular,
performance can degrade under changes in lighting, scale, blur, occlusion,
road appearance, or other distribution shifts.

**Confusion behavior:**  
The model occasionally struggles with severe over-detection on gravel or degraded roads where textures mimic potholes. Missed detections typically occur on very small, distant potholes or those obscured by harsh lighting/glare.

---

## 4. Five Failure Cases

| # | Failure | Root cause |
|---|---|---|
| 1 | img-106 | False positives + poor localization (Texture confusion from dark stains/shadows) |
| 2 | img-206 | Missed pothole (Small scale or partial occlusion) |
| 3 | img-421 | Severe over-detection (4 GT vs 32 predictions due to highly degraded road surface confusing the detector) |
| 4 | img-266 | Extra detections + missed small region (Complex scene with multiple ambiguous road defects) |
| 5 | img-566 | Multiple missed/poorly localized (10 GT vs 15 predictions due to crowded/dense grouping of potholes causing overlapping boxes) |

These failures demonstrate the model's limitations rather than being excluded
from the analysis.

---

## 5. Part B — Minimal Reasoning Layer

The `/ask` endpoint uses a single hand-written decision layer utilizing Groq's `openai/gpt-oss-120b` running in native JSON mode for reasoning. If the Groq API times out, the code gracefully degrades to returning raw detection statistics rather than crashing.

**Routing:**

1. Determine whether the question requires image detection.
2. If not required, answer without invoking RT-DETR.
3. If required, run RT-DETR.
4. Pass structured detections (classes, counts, boxes, confidence) to the
   reasoning layer.
5. Apply the confidence/out-of-scope guardrails before producing the answer.

**Insufficient-information example:**

**Question:** "How deep is this pothole?"

**Result:**  
"Insufficient information: this model only detects pothole presence and location, not attributes like depth, age, or cause."

This prevents the reasoning layer from inventing information that the detector
cannot establish.

---

## 6. API

### `/detect`

Accepts an image and returns detected objects, bounding boxes, and confidence
scores.

**Request:** `curl -X POST "http://localhost:8000/detect" -F "file=@test_img.jpg"`

**Response:** `{"detections":[{"class_name":"pothole","confidence":0.91,"box":{"x1":120,"y1":180,"x2":340,"y2":420}}]}`

### `/ask`

Accepts an image and natural-language question.

**Request:** `curl -X POST "http://localhost:8000/ask?question=How%20many%20potholes%20are%20there?" -F "file=@test_img.jpg"`

**Response:** `{"answer":"The image contains 1 detected pothole.","detections":[{"class_name":"pothole","confidence":0.91,"box":{"x1":120,"y1":180,"x2":340,"y2":420}}]}`
