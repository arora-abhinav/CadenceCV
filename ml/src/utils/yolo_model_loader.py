#Reusable script to load and use the Yolo26x pose estimation model
from ultralytics import YOLO

def load_source(source_dir, model, show: bool = False, conf: float = 0.3, stream: bool = True, imgsz:int = 640):
    #stream=True now allows us to process the video frame by frame
    source = model.predict(source=source_dir, show=show, conf=conf, stream=stream, imgsz = imgsz)

    return source

def load_model(yolo_model:str):
    #stream=True now allows us to process the video frame by frame
    model = YOLO(yolo_model)
    return model