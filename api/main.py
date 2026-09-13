import logging
import time
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from .detector import Detector
from .schemas import DetectionResponse, ErrorResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from .reasoning import answer_question

# Set up structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pothole-api")

app = FastAPI(title="Pothole Detection & Reasoning API")

# Initialize the detector once at startup
# We look for the weights relative to the project root
weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "best.pt")
if not os.path.exists(weights_path):
    # Fallback to the training dir if not in weights
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "runs", "train", "pothole_rtdetr", "weights", "best.pt")
    
detector = Detector(model_path=weights_path)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} status={response.status_code} time={time.time()-start:.3f}s")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500, 
        content={"error": "internal_error", "detail": str(exc)}
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    image_bytes = await file.read()
    return detector.predict(image_bytes)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(question: str, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
        
    image_bytes = await file.read()
    result = answer_question(question, image_bytes, detector)
    return result

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": detector.is_loaded
    }

# Ensure Uvicorn runs when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
