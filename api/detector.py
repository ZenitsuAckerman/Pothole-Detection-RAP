import os
from ultralytics import RTDETR
from PIL import Image
import io
from .schemas import Detection, BoundingBox, DetectionResponse

# Use environment variable for weights path to make it easy to configure via Docker or .env
MODEL_PATH = os.getenv("MODEL_PATH", "../training/runs/train/pothole_rtdetr/weights/best.pt")
CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.25))

class Detector:
    def __init__(self, model_path: str = MODEL_PATH):
        # We handle the case where weights might not exist yet gracefully for the initial API scaffolding
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = RTDETR(model_path)
            self.is_loaded = True
        else:
            print(f"Warning: Model weights not found at {model_path}. Detection will be disabled.")
            self.model = None
            self.is_loaded = False

    def predict(self, image_bytes: bytes) -> DetectionResponse:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if not self.is_loaded:
            return DetectionResponse(detections=[], image_width=img.width, image_height=img.height)
            
        results = self.model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=self.model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
            ))

        return DetectionResponse(
            detections=detections,
            image_width=img.width,
            image_height=img.height,
        )
