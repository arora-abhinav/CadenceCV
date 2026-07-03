#Reusable script to load and use the Yolo26x pose estimation model
from ultralytics import YOLO

def load_source(source_dir, show: bool, conf: float, stream: bool, imgsz:int, model):
    #stream=True now allows us to process the video frame by frame
    source = model.predict(source=source_dir, show=False, conf=0.3, stream=True, imgsz = 1280)

    return source

def load_model(yolo_model:str):
    #stream=True now allows us to process the video frame by frame
    model = YOLO(yolo_model)
    return model