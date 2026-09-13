import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)

@pytest.fixture
def dummy_image_bytes():
    """Create a valid dummy JPEG in memory."""
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "model_loaded" in data

def test_detect_endpoint(dummy_image_bytes):
    # Using the dummy image since no real image is guaranteed in the repo
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "image_width" in data
    assert "image_height" in data
    
    # Verify structure of detections if any exist
    for det in data["detections"]:
        assert "class_name" in det
        assert "confidence" in det
        assert "box" in det

def test_ask_detection_path(dummy_image_bytes, monkeypatch):
    """Test the /ask endpoint detection path with mocked Groq call."""
    
    # Mock the LLM call in reasoning.py
    def mock_call_llm(prompt, max_tokens, json_mode):
        if json_mode:
            return '{"needs_detection": true}'
        return "This is a mocked answer from the structured reasoner."
        
    monkeypatch.setattr("api.reasoning.call_llm", mock_call_llm)
    
    response = client.post(
        "/ask",
        params={"question": "How many potholes are there?"},
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data.get("used_detection") is True
    assert "answer" in data
    assert "detections" in data
    assert data["answer"] == "This is a mocked answer from the structured reasoner." or data["answer"] == "No potholes were detected in this image with sufficient confidence."

def test_ask_guardrail_out_of_scope(dummy_image_bytes, monkeypatch):
    """Test the /ask endpoint guardrail for out-of-scope questions."""
    
    response = client.post(
        "/ask",
        params={"question": "How deep is this pothole?"},
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data.get("used_detection") is False
    assert data.get("guardrail_triggered") == "out_of_scope"
    assert "Insufficient information" in data.get("answer", "")

def test_invalid_upload():
    """Test that a non-image file returns a 400 error."""
    # Send text content instead of an image
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"This is not an image", "text/plain")}
    )
    assert response.status_code == 400
