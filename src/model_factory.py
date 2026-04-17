import onnxruntime as ort
import numpy as np
import cv2

class ModelFactory:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # The 80 names from the COCO dataset YOLO was trained on
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def predict(self, image_path):
        # 1. Preprocess & keep track of original size
        raw_img = cv2.imread(image_path)
        orig_h, orig_w = raw_img.shape[:2]
        
        img = cv2.resize(raw_img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # 2. Run Inference
        outputs = self.session.run(None, {self.input_name: img})
        predictions = np.squeeze(outputs[0]).T # (8400, 84)

        boxes, scores, class_ids = [], [], []

        # 3. Filter and Scale Coordinates
        for row in predictions:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            conf = classes_scores[class_id]
            
            if conf > 0.4:
                x, y, w, h = row[0:4]
                # Scale from 640 back to original
                x1 = int((x - w/2) * (orig_w / 640))
                y1 = int((y - h/2) * (orig_h / 640))
                width = int(w * (orig_w / 640))
                height = int(h * (orig_h / 640))
                
                boxes.append([x1, y1, width, height])
                scores.append(float(conf))
                class_ids.append(class_id)

        # 4. NMS (Remove duplicates)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
        
        for i in indices:
            x, y, w, h = boxes[i]
            label = f"{self.classes[class_ids[i]]}: {scores[i]:.2f}"
            # Draw green box and text
            cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(raw_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return raw_img