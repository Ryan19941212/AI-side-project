from ultralytics import YOLO

# 載入輕量模型
model = YOLO('yolov8n.pt')

# 訓練
model.train(data='data.yaml', epochs=20, imgsz=640)

# 儲存訓練成果
model.export(format='onnx')  # 可選：轉成 onnx 推論用
