import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, "/media/nvidia/1337480C3244A7BA/MTFENet")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/mtfenet-s.yaml', task='multi')  # build a new model from YAML
    # modelpath = r'/media/wangbei/文件/Lane_marking/MTFENet/ultralytics/yolov8n.pt'
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data=r'/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml', batch=8,cache=True,epochs=30, imgsz=(640,640), device='0,1', name='final', val=True, task='multi',classes=[6,7,8,9,10,11],combine_class=[6,7,8,9], single_cls=True)
    # model.train(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml', batch=32,amp=False,cache='disk',epochs=300, imgsz=(640,640), device='0,1', name='A-test', val=True, task='multi',classes=[6,7,8,9,10,11],combine_class=[6,7,8,9], single_cls=True)
    # model.train(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfnet-9.yaml', batch=32,amp=False,cache='disk',epochs=300, imgsz=(640,640), device='0,1', name='A-test', val=True, task='multi',classes=[0,1,2,3,4,5,6,7,8], combine_class=[0,1,2,3,4,5,6],single_cls=True)
    # model.train(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfnet-9.yaml', batch=32,cache='disk',epochs=300, imgsz=(640,640), device='0,1', name='A-test', val=True, task='multi',classes=[0,1,2,3,4,5,6,7,8], combine_class=[0,1,2,3,4,5,6],single_cls=True)
    # model.train(data='/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml', batch=16,cache=True,epochs=300, imgsz=(640,640), device='0,1', name='final', val=True, task='multi',classes=[0,1,2,3,4,5,6,7,8,9,10,11,12],combine_class=[0,1,2,3,4,5,6,7,8,9,10], single_cls=True)






