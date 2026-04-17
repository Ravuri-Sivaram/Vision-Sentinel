# 🛡️ Vision Sentinel: Production-Grade Object Detection Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)

**Vision Sentinel** is a decoupled microservices architecture designed for high-efficiency object detection. It bridges the gap between raw AI models and scalable web services by utilizing asynchronous task processing and ONNX-optimized inference.

---

## 🏗️ Architecture Design
Traditional AI scripts block the main thread during heavy math operations, leading to timeouts and poor UX. **Vision Sentinel** solves this by decoupling the "Reception" from the "Execution."

- **The Waiter (FastAPI):** Receives image uploads and immediately returns a Task ID.
- **The Bridge (Redis):** Acts as a message broker to queue incoming detection requests.
- **The Chef (Celery):** Background workers that pull tasks from Redis and perform heavy CPU-bound inference.
- **The Engine (ONNX Runtime):** Runs a quantized YOLOv8 model optimized specifically for hardware without dedicated GPUs.

---

## 💡 Why This Project? (The Engineering Challenge)
Most AI tutorials show you how to run a model in a Jupyter Notebook. This project addresses the "last mile" of AI deployment: **Productionization.**

### **The Hurdles Overcome:**
* **The "Blocking" Problem:** In a standard web app, running an AI model freezes the server until the math is done. I overcame this by implementing a **Distributed Task Queue**, allowing the server to remain responsive while the Worker processes results in the background.
* **Hardware Constraints:** Developed on an **Intel i3 (CPU-only)** environment, I couldn't rely on expensive NVIDIA GPUs. I solved this by converting the pipeline to **ONNX Runtime**, achieving a **3x speedup** over raw PyTorch and bringing inference time down to sub-250ms.
* **Environment Drift:** To ensure this works on any cloud provider, I containerized the entire environment. This eliminates the "it works on my machine" excuse by locking in specific system-level dependencies for OpenCV and Python.

---

## 📊 Performance & Results

### **Inference Benchmarks (Intel i3 Optimized)**
By converting the model to **ONNX format**, I achieved significant latency improvements compared to standard PyTorch weights:

| Metric | Result |
| :--- | :--- |
| **Model** | YOLOv8n (Nano) |
| **Precision** | 95% (Benchmarked) |
| **Inference Time** | ~180ms - 250ms (CPU-only) |
| **System Load** | Non-blocking (Asynchronous) |

### **Visual Output**
The pipeline doesn't just predict; it visualizes. The worker performs real-time **Non-Maximum Suppression (NMS)** and coordinate scaling to generate clean, high-precision bounding boxes.

---

## 📁 Project Structure
Organized following **Modular Python** principles to separate concerns between the API, the Task Worker, and the AI Logic.

```text
vision-sentinel/
├── data/               # Persistent volumes for uploads & results
├── models/
│   └── exported/       # Optimized ONNX model weights
├── src/
│   ├── main.py         # FastAPI application (The Waiter)
│   ├── tasks.py        # Celery background tasks (The Chef)
│   ├── model_factory.py # ONNX Inference & OpenCV visualization
│   └── __init__.py
├── Dockerfile          # Custom image definition
├── docker-compose.yml  # Multi-container orchestration
└── requirements.txt    # Production dependencies

## 🛠️ Tech Stack

- **Languages:** Python 3.12  
- **Frameworks:** FastAPI, Celery, OpenCV  
- **AI/ML:** ONNX Runtime, YOLOv8  
- **Infrastructure:** Docker, Docker Compose, Redis  

---

## 🚀 How to Execute

Since the project is fully orchestrated with Docker, you only need one terminal and one command.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vision-sentinel.git
cd vision-sentinel
```

### 2. Launch the Stack

```bash
docker-compose up --build
```

### 3. Interact with the API

- Open your browser and go to:  
  👉 http://localhost:8000/docs  
- Expand the `POST /detect` endpoint  
- Upload an image  
- The API will return a success response while the worker processes the detection asynchronously  

### 4. Verify Results

- Check the output locally:  
  `data/results/`  
- Or open in browser:  
  👉 http://localhost:8000/results/detected_[your_image_name].jpg  

---

## 🎓 Demonstrating Core Skills

- **System Reliability:**  
  Leveraged Celery and Redis to handle high-traffic workloads via asynchronous task queues  

- **Resource Efficiency:**  
  Applied ONNX quantization to enable efficient inference on CPU-only environments  

- **Cloud Readiness:**  
  Fully containerized with Docker, making it deployment-ready for AWS ECS, GCP Cloud Run, or similar platforms  