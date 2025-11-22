import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("model/yolov8n.pt")

def extract_features(input_data):

    # CASE 1: input is a numpy image (frame from video)
    if isinstance(input_data, np.ndarray):
        img = input_data

    # CASE 2: input is a file path
    else:
        img = cv2.imread(input_data)

    if img is None:
        return None

    results = model.predict(img, verbose=False)

    features = []
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]

            try:
                resized = cv2.resize(cropped, (128, 128))
                feat = resized.flatten() / 255.0
                features.append(feat)
            except:
                pass

    if len(features) == 0:
        resized = cv2.resize(img, (128, 128))
        return resized.flatten() / 255.0

    return np.mean(features, axis=0)
