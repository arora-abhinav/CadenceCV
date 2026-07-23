#This will be the inference file for the support vector machine
import pickle
from sklearn.pipeline import make_pipeline, Pipeline
import os
from ultralytics import YOLO
import torch
d_frame_model_string = "d_frame_svm.pkl"
u_frame_model_string = "u_frame_svm.pkl"
weights_dir = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights"
test_video_dir = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4"

#Type casting to get access to methods
with open(os.path.join(weights_dir, d_frame_model_string), "rb") as file:
    d_frame_model:Pipeline = pickle.load(file)

with open(os.path.join(weights_dir, u_frame_model_string), "rb") as file:
    u_frame_model:Pipeline = pickle.load(file)

def obtain_normalised_keypoints(video_dir):
    model = YOLO("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/21_keypoints.pt")
    preds = model.predict(video_dir, imgsz=1280, stream=False)
    all_kpts = []
    for res in preds:
        if len(res) == 0:
            continue
        #The highest confidence prediction keypoints
        kpts = res.keypoints.xy[0]
        #Corresponding person's bounding boxes
        bbox = res.boxes.xyxy[0]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        x_diff = x2 - x1
        y_diff = y2 - y1
        kpts_copy = kpts.copy_(kpts_copy)
        for (index, k) in enumerate(kpts):
            k.tolist()
            x = (x - x1)/x_diff
            y = (y - y1)/y_diff
            kpts_copy[index] = [x, y]
    
        all_kpts.append(kpts_copy)
    
    #Flattening to a list since thats required by the SVM
    return torch.tensor(all_kpts).flatten(1).tolist()

normalised_kpts_string = "normalised_keypoints.pkl"
kpts_path = os.path.join(weights_dir, normalised_kpts_string)
if not os.path.isfile(kpts_path):
    kpts = obtain_normalised_keypoints(test_video_dir)
    with open(kpts_path, "wb") as file:
        pickle.dump(kpts, file)

else:
    with open(kpts_path, "rb") as file:
        kpts = pickle.load(file)

u_frame_preds = u_frame_model.predict(kpts)
d_frame_preds = d_frame_model.predict(kpts)