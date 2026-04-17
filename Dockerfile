# Use a lightweight Python base
FROM python:3.12-slim

# Install system libraries required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this speeds up builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and models
COPY . .

# Ensure Python can find your 'src' folder
ENV PYTHONPATH=/app

ENV PYTHONUNBUFFERED=1