from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(source='data/images/train/sample.jpg', save=True, show=True)
