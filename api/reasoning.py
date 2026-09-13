import json
import logging
import os
from .llm_client import call_llm

logger = logging.getLogger("pothole-api")

CONF_GUARDRAIL_THRESHOLD = float(os.getenv("CONF_GUARDRAIL_THRESHOLD", 0.4))
OUT_OF_SCOPE_KEYWORDS = ["depth", "how old", "cause", "when was", "material", "brand"]
DETECTION_KEYWORDS = ["how many", "is there", "count", "safe", "visible",
                      "present", "any pothole", "damage", "potholes"]

def intent_router(question: str) -> bool:
    """Rule-based first; falls back to a single constrained LLM call only if ambiguous."""
    q = question.lower()
    
    # Rule-based fast paths
    if any(k in q for k in DETECTION_KEYWORDS):
        return True
    if any(k in q for k in ["what is", "who", "capital of", "weather"]):
        return False

    # Ambiguous — one explicit, constrained LLM call in JSON mode
    prompt = (
        f'Question: "{question}"\n'
        "Does answering this question require analyzing objects detected in an image "
        "(counts, presence, positions of potholes)? "
        'Respond with ONLY valid JSON containing a single boolean key: {"needs_detection": true} or {"needs_detection": false}'
    )
    
    raw = call_llm(prompt, max_tokens=30, json_mode=True)
    
    if raw is None:
        # LLM failure -> fail safe toward running detection rather than guessing blind
        logger.warning("intent_router LLM call failed. Defaulting to needs_detection=True")
        return True
        
    try:
        return json.loads(raw.strip())["needs_detection"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse intent_router JSON response: {raw}, error: {e}")
        return True

def is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in OUT_OF_SCOPE_KEYWORDS)

def structured_reasoner(question: str, detection_response) -> str:
    summary = {
        "count": len(detection_response.detections),
        "confidences": [round(d.confidence, 2) for d in detection_response.detections],
        "boxes": [d.box.dict() for d in detection_response.detections],
        "image_size": [detection_response.image_width, detection_response.image_height],
    }
    prompt = (
        f"You are answering a question about an image using ONLY this structured "
        f"detection output (no other knowledge of the image):\n{json.dumps(summary)}\n\n"
        f'Question: "{question}"\n'
        "Answer in one or two plain-language sentences, grounded strictly in the data above."
    )
    
    answer = call_llm(prompt, max_tokens=150, json_mode=False)
    
    if answer is None:
        # Graceful degradation if LLM is unavailable
        logger.warning("structured_reasoner LLM call failed. Gracefully degrading response.")
        num_potholes = len(detection_response.detections)
        max_conf = max((d.confidence for d in detection_response.detections), default=0)
        return f"Insufficient information: the reasoning service is temporarily unavailable, but detection found {num_potholes} pothole(s) with max confidence {max_conf:.2f}."
        
    return answer

def answer_question(question: str, image_bytes: bytes, detector) -> dict:
    needs_detection = intent_router(question)

    if not needs_detection:
        answer = call_llm(f"Answer briefly: {question}", max_tokens=100, json_mode=False)
        return {
            "answer": answer if answer else "The reasoning service is temporarily unavailable.",
            "used_detection": False,
        }

    if is_out_of_scope(question):
        logger.info(f"Guardrail triggered: out_of_scope for question '{question}'")
        return {
            "answer": "Insufficient information: this model only detects pothole "
                      "presence and location, not attributes like depth, age, or cause.",
            "used_detection": False,
            "guardrail_triggered": "out_of_scope",
        }

    detection_response = detector.predict(image_bytes)

    if len(detection_response.detections) == 0:
        return {
            "answer": "No potholes were detected in this image with sufficient confidence.",
            "used_detection": True,
            "detections": [],
        }

    max_conf = max(d.confidence for d in detection_response.detections)
    if max_conf < CONF_GUARDRAIL_THRESHOLD:
        logger.info(f"Guardrail triggered: low_confidence ({max_conf:.2f} < {CONF_GUARDRAIL_THRESHOLD})")
        return {
            "answer": "Insufficient information: detections in this image fall below "
                      f"the confidence threshold ({max_conf:.2f} < {CONF_GUARDRAIL_THRESHOLD}), "
                      "so I can't answer confidently.",
            "used_detection": True,
            "guardrail_triggered": "low_confidence",
            "max_confidence": max_conf,
        }

    answer = structured_reasoner(question, detection_response)
    return {
        "answer": answer,
        "used_detection": True,
        "detections": [d.dict() for d in detection_response.detections],
    }
