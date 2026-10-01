#Shared YOLO -> bbox normalised keypoints step. This used to live inside svm_inference.py, but the LSTM inference
#needs the exact same keypoints and importing svm_inference runs the whole SVM pipeline (and its prompts) on import.
#So its pulled out here and both svm_inference.py and footnet_inference.py call it
import os
import sys
import pickle
import numpy as np
import torch
from ultralytics import YOLO
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
#single-function left/right correction - clean, mostly-consistent keypoints so the temporal features arent
#poisoned by YOLO's flickering (a flip = a giant fake velocity spike in the per-leg FootNet features)
from side_correction import correct_side

yolo_weights = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/final_model.pt"


def obtain_normalised_keypoints(video_dir):
    model = YOLO(yolo_weights)
    preds = model.predict(video_dir, imgsz=1280, stream=True)
    all_kpts = []
    non_normalised_kpts = []
    valid_frames = []
    bboxes = []
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
        #box needed so correct_side can un-normalise the kpts back to pixels for the marking display
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])

    all_kpts = torch.tensor(all_kpts)
    non_normalised_kpts = torch.tensor(non_normalised_kpts)

    #Flattening to a list since thats required by the SVM
    return {"Normalised Keypoints": all_kpts.flatten(1).tolist(), "Valid Frames": valid_frames,
            "Non-Normalised Keypoints": non_normalised_kpts, "Bounding Boxes": bboxes}


#Runs YOLO + the side correction once per video and caches it, since YOLO at imgsz=1280 is slow and the side
#correction needs the user to mark an anchor frame. Same flow svm_inference.py had at module level.
#The video path is saved in the cache now too - the old cache had no idea which video it came from, so pointing
#it at a new video would silently reuse the old video's keypoints
def load_corrected_keypoints(video_dir, kpts_path):
    if os.path.isfile(kpts_path):
        with open(kpts_path, "rb") as file:
            results = pickle.load(file)
        #Caches made before this change have no "Video" key, those are trusted as is (same as before)
        if results.get("Video", video_dir) == video_dir:
            return results

    results = obtain_normalised_keypoints(video_dir)
    valid_frames = results["Valid Frames"]
    bbox = results["Bounding Boxes"]
    #correct YOLO's flickering left/right BEFORE saving (you mark the anchor). the FootNet temporal features
    #are built from these coords, and a flip = a huge fake velocity spike - correcting kills that noise.
    normalised = np.array(results["Normalised Keypoints"]).reshape(len(valid_frames), 21, 2)
    corrected = correct_side(video_dir, bbox, normalised)                      # (n, 21, 2)
    results["Normalised Keypoints"] = corrected.reshape(len(valid_frames), -1).tolist()
    #re-derive the pixel keypoints from the corrected normalised ones so the visualiser stays in sync
    bbox_arr = np.array(bbox)
    xy1 = bbox_arr[:, None, :2]; wh = bbox_arr[:, None, 2:] - bbox_arr[:, None, :2]
    results["Non-Normalised Keypoints"] = torch.tensor(xy1 + corrected * wh)
    results["Video"] = video_dir
    with open(kpts_path, "wb") as file:
        pickle.dump(results, file)
    return results
