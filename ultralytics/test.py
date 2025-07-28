from ultralytics import YOLO

if __name__=='__main__':
    model=YOLO("/media/nvidia/1337480C3244A7BA/MTFENet/test_yaml/bdd-mtfenet-multi.yaml")
    model.info()