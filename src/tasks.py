from celery import Celery
from src.model_factory import ModelFactory
import cv2
import os


app = Celery('vision_tasks', broker='redis://redis:6379/0')
model = ModelFactory("models/exported/yolov8n.onnx")

@app.task
def process_detection(image_path):
    print(f"Drawing results for: {image_path}")
    
    # Run prediction (now returns an image with boxes)
    result_img = model.predict(image_path)
    
    # Create the results folder if it doesn't exist
    os.makedirs("data/results", exist_ok=True)
    
    # Save the result
    filename = os.path.basename(image_path)
    save_path = os.path.join("data/results", f"detected_{filename}")
    cv2.imwrite(save_path, result_img)
    
    print(f"Result saved at: {save_path}")
    return True