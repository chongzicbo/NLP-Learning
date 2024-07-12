from ultralytics import YOLO

image_path = "./test.png"  # 待预测图片路径
model_path = "/data/bocheng/pretrained_model/360LayoutAnalysis/paper-8n.pt"  # 权重路径
model = YOLO(model_path)

result = model(image_path, save=True, conf=0.5, save_crop=False, line_width=2)
# print(result)
print("\n================================")
print(result[0].names)  # 输出id2label map
# print(result[0].boxes)  # 输出所有的检测到的bounding box
print(result[0].boxes.xyxy)  # 输出所有的检测到的bounding box的左上和右下坐标
print(result[0].boxes.cls)  # 输出所有的检测到的bounding box类别对应的id
# print(result[0].boxes.conf)  # 输出所有的检测到的bounding box的置信度
