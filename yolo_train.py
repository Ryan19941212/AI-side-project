from ultralytics import YOLO

# 載入 YOLOv8 預訓練模型（nano 最輕量，訓練快）
model = YOLO('yolov8n.pt')

# 訓練模型（你可以依需求調整 epochs）
model.train(
    data='data.yaml',     # 指定配置檔路徑
    epochs=20,            # 訓練輪數，先用 20 試看看
    imgsz=640,            # 影像大小
    batch=8,              # 一次丟幾張進模型
    device=0,             # 使用 GPU 訓練
    name='crack500_train' # 訓練結果資料夾名稱
)
