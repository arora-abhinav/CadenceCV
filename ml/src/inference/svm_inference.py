#This will be the inference file for the support vector machine
import pickle
from sklearn.pipeline import make_pipeline, Pipeline
import os
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
d_frame_model_string = "d_frame_svm.pkl"
u_frame_model_string = "u_frame_svm.pkl"
weights_dir = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights"
test_video_dir = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4"

#Type casting to get access to methods
with open(os.path.join(weights_dir, d_frame_model_string), "rb") as file:
    unpickler = pickle.Unpickler(file)
    d_frame_model:Pipeline = unpickler.load()

with open(os.path.join(weights_dir, u_frame_model_string), "rb") as file:
    u_frame_model:Pipeline = pickle.load(file)

def obtain_normalised_keypoints(video_dir):
    model = YOLO("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/final_model.pt")
    preds = model.predict(video_dir, imgsz=1280, stream=True)
    all_kpts = []
    non_normalised_kpts = []
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
        kpts_copy = kpts.clone().tolist()
        #Renamed to kp_index so it doesnt clobber the outer `index` (the actual frame number)
        for (kp_index, k) in enumerate(kpts):
            k.tolist()
            x,y = k
            x = (x - x1)/x_diff
            y = (y - y1)/y_diff
            kpts_copy[kp_index] = [x, y]
    
        all_kpts.append(kpts_copy)
        valid_frames.append(index)
        non_normalised_kpts.append(kpts.tolist())
    
    all_kpts = torch.tensor(all_kpts)
    non_normalised_kpts = torch.tensor(non_normalised_kpts)

    #Flattening to a list since thats required by the SVM
    return {"Normalised Keypoints": all_kpts.flatten(1).tolist(), "Valid Frames": valid_frames,
            "Non-Normalised Keypoints": non_normalised_kpts}

normalised_kpts_string = "normalised_keypoints.pkl"
kpts_path = os.path.join(weights_dir, normalised_kpts_string)
if not os.path.isfile(kpts_path):
    results = obtain_normalised_keypoints(test_video_dir)
    with open(kpts_path, "wb") as file:
        pickle.dump(results, file)
    flattened_kpts = results["Normalised Keypoints"]
    valid_frames = results["Valid Frames"]
    non_normal_kpts = results["Non-Normalised Keypoints"]

else:
    with open(kpts_path, "rb") as file:
        results = pickle.load(file)
    flattened_kpts = results["Normalised Keypoints"]
    valid_frames = results["Valid Frames"]
    non_normal_kpts:torch.Tensor = results["Non-Normalised Keypoints"]


u_frame_preds = u_frame_model.predict(flattened_kpts)
d_frame_preds = d_frame_model.predict(flattened_kpts)

u_frame_probabilities = u_frame_model.predict_proba(flattened_kpts)
d_frame_probablities = d_frame_model.predict_proba(flattened_kpts)

#This will take the valid_frames from keypoints which will be used to visualise those keypoint coordinates
def visualise_frames(video_dir, frames, preds):
    cap = cv2.VideoCapture(video_dir)

    for pred in preds:
        #pred is the position of the detected event in valid_frames, so frames[pred] is its real frame number
        frame_number = int(frames[pred])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = cap.read()
        #Skip if the frame couldnt be read (otherwise cvtColor crashes on None)
        if not ret:
            continue
        #cv2 uses BGR, matplotlib expects RGB
        cv2.imshow("Event", img)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

#Some noise needs to be filtered out since back to back predictions are being made 
# which is impossible for either u or d frames given the fact that these frames have only 1 occurence per cycle
# Therefore, probablities will be used instead to keep the 
#highest probability classification. The second column is the Yes Preds and the first column is the No Preds

def clean_preds(preds, probabilities, tag):
    correct_preds = np.array(np.where(np.array(preds) == 1)).flatten().tolist()
    indices = []
    current_block = []
    i = 0
    j = 0
    #This probably could've been done using numpy np.diff and np.where and whatnot but this seemed more explicit and thus
    #better
    while i + j < len(correct_preds):
        if correct_preds[i] + j == correct_preds[i + j]:
            current_block.append(correct_preds[i + j])
        else:
            indices.append(current_block.copy())
            current_block.clear()
            i += j
            j = 0
            continue
        j += 1

    to_keep_indices = []
    for group in indices:
        grp_probs = []
        for ind in group:
            grp_probs.append(probabilities[ind][1])
        i = np.array(np.where(np.array(grp_probs) == max(grp_probs))).flatten().tolist()[0]
        to_keep_indices.append(group[i])

    return to_keep_indices

cleaned_u_preds = clean_preds(u_frame_preds, u_frame_probabilities.tolist(), "U")
cleaned_d_preds = clean_preds(d_frame_preds, d_frame_probablities.tolist(), "D")

consecutive_preds = cleaned_u_preds + cleaned_d_preds
consecutive_preds.sort()

visualise_frames(test_video_dir, valid_frames, cleaned_d_preds)