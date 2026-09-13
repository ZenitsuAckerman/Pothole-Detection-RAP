# Problem Statement Memo: Pothole Detection & Reasoning API

## 1. Domain & Dataset
- **Domain:** Road Infrastructure Safety / Pothole Detection
- **Source:** Roboflow Universe (`yolo-sfvlm/pothole-detection-using-yolov5-p20qq`)
- **License:** MIT
- **Justification:** I selected "potholes" because it is a vital real-world CV application (preventing vehicle damage and accidents) and strictly satisfies the constraint of being a non-COCO class.
- **Architecture Note:** Dataset labels are stored in YOLO-format annotation files (`data.yaml` + `.txt` boxes) because that is the export convention Roboflow and Ultralytics share across all detectors. The model trained is RT-DETR (`ultralytics.RTDETR("rtdetr-l.pt")`, a transformer-based detector), not a YOLO architecture. Some source dataset titles contain "yolo" in the name because that's what the original uploader exported for — this refers only to the label file format, not the model used here.

## 2. Dataset Curation & Scope Limitation
- Excluded N boxes < 0.1% area — this is a stated scope limitation, not a quality claim. See failure case #4 for how the model behaves on this excluded range.
- **Honesty/Trade-off:** We excluded N boxes under 0.1% image area from the primary training run because inspection showed several were likely mislabeled. This is a real scope limitation: our reported metrics do not reflect performance on very small/distant potholes, and we treat this as one of our five failure cases below, not as a quality improvement.

## 3. Split Strategy
- We maintained the original train/val/test split provided by the dataset author (Train: X%, Val: Y%, Test: Z%). 
- We evaluated the model specifically on the **test split** for our final reported metrics to ensure the numbers reflect genuine generalization, strictly separating them from the `val` split which was used for early stopping (patience=15).

## 4. Evaluation Metrics
*Metrics computed via `evaluation/eval.py`.*
- **Val mAP50/50-95 (tuning signal):** from `val_metrics.json`
- **Test mAP50/50-95 (headline, honest proxy for hidden set):** from `test_metrics.json`
- **Gap between val and test:** [discuss what it implies about hidden-set risk, e.g., slightly lower test metrics indicate the val set was easier and the hidden set may be tougher still.]

## 5. Five Failure Cases
*(Mined automatically from test split — see `evaluation/failure_cases/`)*
1. **False Positive (Shadow/Stain):** [Root Cause - e.g. model confused dark patch for pothole]
2. **False Negative (Water-filled):** [Root Cause - e.g. glare destroyed texture]
3. **Low-confidence borderline:** [Root Cause - e.g. partially occluded by tire]
4. **Small/distant pothole (from excluded-box population):** [Root Cause - falls outside trained scale distribution because we excluded < 0.1% area boxes]
5. **Class confusion (Crack vs Pothole) / blur / lighting:** [whichever test-split example is worst]

## 6. Part B — Reasoning Layer
Our reasoning layer utilizes Groq's `llama-3.3-70b-versatile` running in native JSON mode for intent routing. 
- **Routing:** A rule-based regex catches obvious queries. Ambiguous queries trigger the LLM to output `{"needs_detection": bool}`. 
- **Fail-Safe:** If the Groq API times out, the code gracefully degrades to returning raw detection statistics rather than crashing with a 500 error.

### Insufficient Info Example
*Pulled from actual guardrail logs at `evaluation/reasoning_logs.txt`:*
- **Query:** "How deep is this pothole?" 
- **Guardrail:** `out_of_scope` guardrail triggered.
- **Result:** "Insufficient information: this model only detects pothole presence and location, not attributes like depth, age, or cause."
