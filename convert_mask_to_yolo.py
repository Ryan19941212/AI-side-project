import os
import cv2

image_dir = 'CRACK500/images'
mask_dir = 'CRACK500/masks'
output_image_dir = 'data/images/train'
output_label_dir = 'data/labels/train'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for filename in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, filename)
    image_path = os.path.join(image_dir, filename.replace('_mask.png', '.jpg'))
    if not os.path.exists(image_path): continue

    mask = cv2.imread(mask_path, 0)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    label_txt = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_norm = bw / w
        bh_norm = bh / h
        label_txt.append(f"0 {x_center} {y_center} {bw_norm} {bh_norm}")

    with open(os.path.join(output_label_dir, filename.replace('.png', '.txt')), 'w') as f:
        f.write('\n'.join(label_txt))

    cv2.imwrite(os.path.join(output_image_dir, filename.replace('.png', '.jpg')), img)

print("✅ 已成功轉換為 YOLO 格式")
