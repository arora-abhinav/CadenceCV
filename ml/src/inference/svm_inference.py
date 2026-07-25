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


u_frame_probabilities = u_frame_model.predict_proba(flattened_kpts)
d_frame_probablities = d_frame_model.predict_proba(flattened_kpts)

#The u model is under-confident about toe-offs (its positive probabilities sit lower than the d model's),
#so the default 0.5 cutoff drops ~half the real toe-offs. Thresholding the probability at ~0.42 catches them.
#Tune this so the toe-off count matches the strikefoot count (they pair 1:1 per leg). d is fine on the default.
u_frame_preds = (u_frame_probabilities[:, 1] > 0.43).astype(int)
d_frame_preds = d_frame_model.predict(flattened_kpts)

print(len(d_frame_preds))

#This will take the valid_frames from keypoints which will be used to visualise those keypoint coordinates
#keypoints here are the NON normalised ones (raw pixel xy from YOLO) so they land in the right spot on the frame.
#Im drawing + naming the foot keypoints specifically because im trying to see with my own eyes whether YOLO is
#putting "left toe/heel" on the actual left foot or if its swapping them - which would explain the leg mislabelling
def visualise_frames(video_dir, frames, preds, keypoints):
    cap = cv2.VideoCapture(video_dir)

    #The indices im pulling in obtain_which_leg, named so i can read them off the image
    foot_names = {17: "left toe", 18: "right toe", 19: "left heel", 20: "right heel"}

    for pred in preds:
        #pred is the position of the detected event in valid_frames, so frames[pred] is its real frame number
        frame_number = int(frames[pred])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = cap.read()
        #Skip if the frame couldnt be read (otherwise cvtColor crashes on None)
        if not ret:
            continue

        #keypoints[pred] = this frames 21 pixel keypoints, same index order YOLO gives them
        pose = keypoints[pred].tolist() if hasattr(keypoints[pred], "tolist") else keypoints[pred]
        for idx, (x, y) in enumerate(pose):
            x, y = int(x), int(y)
            #Every keypoint as a small green dot just for context
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            #The 4 foot ones get a bigger red dot + their name so left/right is obvious at a glance
            if idx in foot_names:
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, foot_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #cv2 uses BGR, matplotlib expects RGB
        cv2.imshow("Event", img)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

#Some noise needs to be filtered out since back to back predictions are being made 
# which is impossible for either u or d frames given the fact that these frames have only 1 occurence per cycle
# Therefore, probablities will be used instead to keep the 
#highest probability classification. The second column is the Yes Preds and the first column is the No Preds

def clean_preds(preds, probabilities):
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

    #The while loop exits without appending the final block, so add it here
    #of every video gets silently dropped
    if current_block:
        indices.append(current_block.copy())

    to_keep_indices = []
    for group in indices:
        grp_probs = []
        for ind in group:
            grp_probs.append(probabilities[ind][1])
        i = np.array(np.where(np.array(grp_probs) == max(grp_probs))).flatten().tolist()[0]
        to_keep_indices.append(group[i])

    return to_keep_indices

cleaned_u_preds_indices = clean_preds(u_frame_preds, u_frame_probabilities.tolist())
cleaned_d_preds_indices = clean_preds(d_frame_preds, d_frame_probablities.tolist())

print(cleaned_d_preds_indices)

#If runner direction is right, and task is D and right_ankle_coord_x > left_ankle_coord_x, then right leg, else left leg.
#If runner direction is left, and task is D and right_ankle_coord_x > left_ankle_coord_x, then left leg, else right_leg.
#If runner direction is right, and task is U and right_ankle_coord_x > left_ankle_coord_x, then left leg, else right leg.
#If runner direction is left, and task is U and right_ankle_coord_x > left_ankle_coord_x, them right leg, else left leg.
np_valid = np.array(valid_frames)
def obtain_which_leg(cleaned_pred_indices, keypoints, task):
    pred_to_leg_dict = {}
    for p in cleaned_pred_indices:
        pose_kpts = keypoints[p]
        left_heel = np.array(pose_kpts[19])
        left_toe = np.array(pose_kpts[17])
        right_heel = np.array(pose_kpts[20])
        left_knee = np.array(pose_kpts[13])
        left_hip = np.array(pose_kpts[11])
        runner_direction = "right" if (left_knee[0] - left_hip[0]) > 0 else "right"
        if runner_direction == "right":
            if task == "D":
                if right_heel[0] > left_heel[0]:
                    pred_to_leg_dict[p] = "R"
                else:
                    pred_to_leg_dict[p] = "L"
            elif task == "U":
                if right_heel[0] > left_heel[0]:
                    pred_to_leg_dict[p] = "L"
                else:
                    pred_to_leg_dict[p] = "R"

        elif runner_direction == "left":
            if task == "D":
                if right_heel[0] > left_heel[0]:
                    pred_to_leg_dict[p] = "L"
                else:
                    pred_to_leg_dict[p] = "R"
            elif task == "U":
                if right_heel[0] > left_heel[0]:
                    pred_to_leg_dict[p] = "R"
                else:
                    pred_to_leg_dict[p] = "L"
    return pred_to_leg_dict

kpts_reshaped = torch.tensor(flattened_kpts).reshape([253, 21, 2]).tolist()

d_frame_preds_legs_indices = obtain_which_leg(cleaned_d_preds_indices, kpts_reshaped, "D")
u_frame_preds_legs_indices = obtain_which_leg(cleaned_u_preds_indices, kpts_reshaped, "U")
cap = cv2.VideoCapture(test_video_dir)

if __name__ == "__main__":
    visualise_frames(test_video_dir, valid_frames, cleaned_d_preds_indices, non_normal_kpts)