import sys
sys.path.insert(0, "/media/nvidia/1337480C3244A7BA/MTFENet")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# model = YOLO('yolov8s-seg.pt')
# number = 3 #input how many tasks in your work
model = YOLO('/media/nvidia/1337480C3244A7BA/MTFENet/runs/multi/final/weights/best.pt')  # 加载自己训练的模型# Validate the model

metrics = model.val(speed=True,data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml',iou=0.6,conf=0.001, imgsz=(640,640), classes=[6,7,8,9,10,11],combine_class=[6,7,8,9], single_cls=True)  # no arguments needed, dataset and settings remembered
# metrics = model.val(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml',device='0,1',task='multi',name='val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[0,1,2,3,4,5,6,7,8,9,10],combine_class=[0,1,2,3,4,5,6,7,8],single_cls=False)  # no arguments needed, dataset and settings remembered
# metrics = model.val(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml', device='0,1',task='multi',name='val',iou=0.6,conf=0.001, imgsz=(640,640), classes=[6,7,8,9,10,11],combine_class=[6,7,8,9], single_cls=True)

# for i in range(number):
#     print(f'This is for {i} work')
#     print(metrics[i].box.map)    # map50-95
#     print(metrics[i].box.map50)  # map50
#     print(metrics[i].box.map75)  # map75
#     print(metrics[i].box.maps)   # a list contains map50-95 of each category