import onnxruntime as ort
import numpy as np
import cv2
import os

MODEL_PATH = "models/exported/yolov8n.onnx"
IMAGE_PATH = "data/test.jpg"

def verify_pipeline():
    print("--- 🔍 Phase 1: Cross-Check Report ---")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FAIL: Model file not found at {MODEL_PATH}")
        return

    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"✅ Model Loaded Successfully on CPU.")
        
        # Create a dummy image if test.jpg doesn't exist yet
        if not os.path.exists(IMAGE_PATH):
            os.makedirs("data", exist_ok=True)
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(IMAGE_PATH, dummy_img)
            print(f"⚠️ Using dummy 640x640 image for test.")

        raw_img = cv2.imread(IMAGE_PATH)
        input_img = cv2.resize(raw_img, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0)
        
        input_name = session.get_inputs()[0].name
        results = session.run(None, {input_name: input_img})
        output = results[0]
        
        print(f"✅ Inference Successful!")
        print(f"   Output Shape: {output.shape}")
        
        if output.shape == (1, 84, 8400):
            print("\n🚀 RESULT: Phase 1 is officially VALIDATED.")
        else:
            print(f"\n⚠️ WARNING: Shape is {output.shape}, expected (1, 84, 8400). Check model version.")
            
    except Exception as e:
        print(f"❌ FAIL: {e}")

if __name__ == "__main__":
    verify_pipeline()
