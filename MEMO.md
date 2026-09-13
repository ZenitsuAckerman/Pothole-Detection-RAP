# Pothole Detection & Reasoning API
### Technical Screening Memo

## 1. Domain & Dataset

**Domain:** Road Infrastructure Safety — Pothole Detection

**Why this domain:**  
Potholes are a practical road-safety and infrastructure-maintenance problem where
localized object detection can support automated road inspection.

**Dataset source:**  
Roboflow Universe — [exact project/version]

**Dataset composition:**  
- Images: [final count]
- Class: `pothole`
- Annotation format: YOLO bounding boxes
- Non-COCO class requirement: satisfied because `pothole` is not a standard COCO class.

**Dataset preparation:**  
[Exactly what we actually did — no invented curation.]

---

## 2. Split Strategy & Training

**Split:**  
- Train: [X]
- Validation: [Y]
- Test: [Z]

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
| Optimizer | `auto` → AdamW |
| Learning rate | 0.002 selected by optimizer |
| Seed | 42 |
| Deterministic | True |
| Hardware | Tesla T4 |
| Training time | [actual time] |

---

## 3. Evaluation & Interpretation

### Results

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Validation | [ ] | [ ] | [ ] | [ ] |
| Test | [ ] | [ ] | [ ] | [ ] |

**Interpretation:**  
[2–4 sentences explaining what the metrics indicate.]

**Limitations:**  
These metrics measure performance on the available evaluation data and do not
guarantee performance on the private hidden evaluation set. In particular,
performance can degrade under changes in lighting, scale, blur, occlusion,
road appearance, or other distribution shifts.

**Confusion behavior:**  
[Actual observations from predictions/confusion analysis.]

---

## 4. Five Failure Cases

| # | Failure | Root cause |
|---|---|---|
| 1 | [actual example] | [actual reason] |
| 2 | [actual example] | [actual reason] |
| 3 | [actual example] | [actual reason] |
| 4 | [actual example] | [actual reason] |
| 5 | [actual example] | [actual reason] |

These failures demonstrate the model's limitations rather than being excluded
from the analysis.

---

## 5. Part B — Minimal Reasoning Layer

The `/ask` endpoint uses a single hand-written decision layer.

**Routing:**

1. Determine whether the question requires image detection.
2. If not required, answer without invoking RT-DETR.
3. If required, run RT-DETR.
4. Pass structured detections (classes, counts, boxes, confidence) to the
   reasoning layer.
5. Apply the confidence/out-of-scope guardrails before producing the answer.

**Insufficient-information example:**

**Question:** "[actual question]"

**Result:**  
"[actual insufficient-information response]"

This prevents the reasoning layer from inventing information that the detector
cannot establish.

---

## 6. API

### `/detect`

Accepts an image and returns detected objects, bounding boxes, and confidence
scores.

**Request:** [sample]

**Response:** [actual sample]

### `/ask`

Accepts an image and natural-language question.

**Request:** [sample]

**Response:** [actual sample]
