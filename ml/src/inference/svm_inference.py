#This will be the inference file for the support vector machine
import pickle
from sklearn.pipeline import make_pipeline, Pipeline
import os
from ultralytics import YOLO
import torch
import cv2
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
    model = YOLO("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/final_model.pt")
    preds = model.predict(video_dir, imgsz=1280, stream=False)
    all_kpts = []
    valid_frames = []
    for index, res in enumerate(preds):
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
            x,y = k
            x = (x - x1)/x_diff
            y = (y - y1)/y_diff
            kpts_copy[index] = [x, y]
    
        all_kpts.append(kpts_copy)
        valid_frames.append(index + 1)
    
    all_kpts = torch.tensor(all_kpts)

    
    #Flattening to a list since thats required by the SVM
    return {"Normalised Keypoints": all_kpts.flatten(1).tolist(), "Valid Frames": valid_frames,
            "Non-Normalised Keypoints": kpts}

normalised_kpts_string = "normalised_keypoints.pkl"
kpts_path = os.path.join(weights_dir, normalised_kpts_string)
if not os.path.isfile(kpts_path):
    results = obtain_normalised_keypoints(test_video_dir)
    with open(kpts_path, "wb") as file:
        pickle.dump(results, file)
    flattened_kpts = results["Normlised Keypoints"]
    valid_frames = results["Valid Frames"]
    bbox = results["Bounding Boxes"]
    non_normal_kpts = results["Non-Normalised Keypoints"]

else:
    with open(kpts_path, "rb") as file:
        results = pickle.load(file)
    flattened_kpts = results["Normalised Keypoints"]
    valid_frames = results["Valid Frames"]
    bbox = results["Bounding Boxes"]
    non_normal_kpts = results["Non-Normalised Keypoints"]


u_frame_preds = u_frame_model.predict(flattened_kpts)
d_frame_preds = d_frame_model.predict(flattened_kpts)

#This will take the valid_frames from keypoints which will be used to visualise those keypoint coordinates
def visualise_frames(video_dir, frames, preds, kpts):
    cap = cv2.VideoCapture(video_dir)
    correct_preds = preds == 1

    for index, pred in enumerate(correct_preds):
        if pred == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[index])
            ret, img = cap.read()
            #This assumes points is a list of coordinates 
            for point in kpts[index]:
                cv2.circle(img, point, 10, thickness=-1, color=(0, 0, 255))
            cv2.imshow(img)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

visualise_frames(test_video_dir, valid_frames, u_frame_preds, non_normal_kpts)