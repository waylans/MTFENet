import sys
sys.path.insert(0, "/media/nvidia/1337480C3244A7BA/MTFENet")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/media/nvidia/1337480C3244A7BA/MTFENet/runs/multi/final/weights/best.pt')  # Validate the model
model.predict(source='/media/nvidia/1337480C3244A7BA/MTFENet/runs/multi/best/', imgsz=(384,672), device= '0,1',name='mtfenet', save=True, conf=0.25, iou=0.45, show_labels=False)

