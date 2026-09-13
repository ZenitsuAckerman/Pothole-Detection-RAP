FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and other C-bindings if necessary
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (respecting .dockerignore)
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
