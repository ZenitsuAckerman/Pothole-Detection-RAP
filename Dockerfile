FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files and weights
COPY api/ ./api/
# In a real build, we'd copy the weights. If they aren't downloaded yet, detector handles it gracefully.
COPY weights/ ./weights/

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "-m", "api.main"]
