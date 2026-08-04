#This will be the inference file for the support vector machine
import pickle
from sklearn.pipeline import make_pipeline, Pipeline
import os
import sys
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#Same FootNet feature builder the models were RETRAINED with - inference must produce the identical
#52-dim vector (42 keypoints + 10 features) or predict_proba rejects it on a shape mismatch
from utils.footnet_svm_features import footnet_features_for_sequence
d_frame_model_string = "d_frame_svm.pkl"
u_frame_model_string = "u_frame_svm.pkl"
weights_dir = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights"
test_video_dir = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4"

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


#TWO copies of the normalised keypoints: one as-is, one with FLIP_IDX applied (whole-body L<->R relabel).
#YOLO's L/R flickers, so a strike/toe-off the SVM misses in one orientation may be caught in the other.
#Run BOTH through BOTH models and UNION the positive frames -> recover missed events (at the cost of extras).
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]


def build_svm_features(seq):
    #seq: (n, 21, 2) -> (n, 52) list = 42 flattened keypoints + 10 FootNet temporal features
    feats = footnet_features_for_sequence(seq)
    return np.concatenate([seq.reshape(len(seq), -1), feats], axis=1).tolist()


kpts_seq = np.array(flattened_kpts).reshape(len(valid_frames), 21, 2)
flipped_seq = kpts_seq[:, FLIP_IDX, :]                                  #the flip_idx'd copy
svm_features_orig = build_svm_features(kpts_seq)
svm_features_flip = build_svm_features(flipped_seq)

#both orientations through both models
d_probs_orig = d_frame_model.predict_proba(svm_features_orig)
d_probs_flip = d_frame_model.predict_proba(svm_features_flip)
u_probs_orig = u_frame_model.predict_proba(svm_features_orig)
u_probs_flip = u_frame_model.predict_proba(svm_features_flip)

#per-orientation positive frames (d on the default 0.5 cutoff, u on the tuned 0.43)
d_orig_pos = np.flatnonzero(d_frame_model.predict(svm_features_orig) == 1)
d_flip_pos = np.flatnonzero(d_frame_model.predict(svm_features_flip) == 1)
u_orig_pos = np.flatnonzero(u_probs_orig[:, 1] > 0.43)
u_flip_pos = np.flatnonzero(u_probs_flip[:, 1] > 0.43)

#UNION the two orientations' positive frames into a final set, then sort into an array
d_union_frames = np.array(sorted(set(d_orig_pos.tolist()) | set(d_flip_pos.tolist())), dtype=int)
u_union_frames = np.array(sorted(set(u_orig_pos.tolist()) | set(u_flip_pos.tolist())), dtype=int)

#feed the union into the existing pipeline: a binary pred array flagged at the union frames, plus the
#per-frame MAX positive-class probability across the two orientations (so clean_preds keeps the strongest)
d_frame_preds = np.zeros(len(valid_frames), dtype=int); d_frame_preds[d_union_frames] = 1
u_frame_preds = np.zeros(len(valid_frames), dtype=int); u_frame_preds[u_union_frames] = 1


def _combined_probs(pa, pb):
    p1 = np.maximum(pa[:, 1], pb[:, 1])
    return np.stack([1 - p1, p1], axis=1)


d_frame_probablities = _combined_probs(d_probs_orig, d_probs_flip)
u_frame_probabilities = _combined_probs(u_probs_orig, u_probs_flip)

#This will take the valid_frames from keypoints which will be used to visualise those keypoint coordinates
#keypoints here are the NON normalised ones (raw pixel xy from YOLO) so they land in the right spot on the frame.
#Im drawing + naming the foot keypoints specifically because im trying to see with my own eyes whether YOLO is
#putting "left toe/heel" on the actual left foot or if its swapping them - which would explain the leg mislabelling
def visualise_frames(video_dir, frames, preds, keypoints):
    cap = cv2.VideoCapture(video_dir)

    #The indices im pulling in obtain_which_leg, named so i can read them off the image
    foot_names = {17: "left toe", 18: "right toe", 19: "left heel", 20: "right heel"}
    #ankles too (15/16) - these are the ACTUAL points obtain_which_leg compares, so seeing where YOLO
    #puts "left ankle" vs "right ankle" is the direct check on the leg-swap
    ankle_names = {15: "left ankle", 16: "right ankle"}
    #knees (13/14) as well, to trace the whole leg's L/R assignment up the limb
    knee_names = {13: "left knee", 14: "right knee"}

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
            #ankles in yellow so they stand apart from the red feet labels
            elif idx in ankle_names:
                cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(img, ankle_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            #knees in orange - a third colour so foot/ankle/knee are each distinguishable
            elif idx in knee_names:
                cv2.circle(img, (x, y), 6, (0, 165, 255), -1)
                cv2.putText(img, knee_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        #cv2 uses BGR, matplotlib expects RGB
        cv2.imshow("Event", img)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

#Some noise needs to be filtered out since back to back predictions are being made 
# which is impossible for either u or d frames given the fact that these frames have only 1 occurence per cycle
# Therefore, probablities will be used instead to keep the 
#highest probability classification. The second column is the Yes Preds and the first column is the No Preds
def clean_preds(preds, probabilities, max_gap=None, gap_frac=0.4):
    """Collapse clusters of positive predictions into one event each.

    Frames within `max_gap` of each other are treated as detections of the
    same event; the frame with the highest positive probability is kept.

    max_gap : int, optional
        Frames of tolerance. If None, derived per-video as
        gap_frac * median inter-detection gap, so it adapts to cadence
        and frame rate instead of assuming 30 fps.
    """
    #The previous system used to be consecutive predictions of 1s are flagged and the highest probability 
    #is kept per cluster. But, it didnt take care of clusters that were nearby each other but not bunched 
    # as a single cluster

    #This is the equivalent of np.where(np.array(preds) == 1).flatten().tolist()
    correct_preds = np.flatnonzero(np.asarray(preds) == 1)
    if correct_preds.size == 0:
        return []

    probs = np.asarray(probabilities)[:, 1]
    gaps = np.diff(correct_preds)

    #If the max_gap isn't specified, then it is calculated based on the median gap size
    if max_gap is None:
        #non adjacent clusters have a difference of more than 1 since a difference of 1 essentially corresponds to predictions in the 
        #same cluster
        between = gaps[gaps > 1]
        #gap_frac is a parameter that must be tuned across various runners. For now, it was selected to be 0.4 for now.
        #This parameter is essentially saying that the tolerance to detect clusters of positive predictions such that it is possible
        #that the next cluster actually shows positive predictions for the other foot but not for the same foot.

        #To tune this, svm inference must run across training videos where the median gap must be identified and compared with the actual
        #gap. Then the average of those ratios will be taken for max_gap
        max_gap = max(1, int(gap_frac * np.median(between))) if between.size else 1

    # split wherever the gap exceeds tolerance
    groups = np.split(correct_preds, np.flatnonzero(gaps > max_gap) + 1)

    return [int(g[np.argmax(probs[g])]) for g in groups]

cleaned_u_preds_indices = clean_preds(u_frame_preds, u_frame_probabilities.tolist())
cleaned_d_preds_indices = clean_preds(d_frame_preds, d_frame_probablities.tolist())

#If runner direction is right, and task is D and right_ankle_coord_x > left_ankle_coord_x, then right leg, else left leg.
#If runner direction is left, and task is D and right_ankle_coord_x > left_ankle_coord_x, then left leg, else right_leg.
#If runner direction is right, and task is U and right_ankle_coord_x > left_ankle_coord_x, then left leg, else right leg.
#If runner direction is left, and task is U and right_ankle_coord_x > left_ankle_coord_x, them right leg, else left leg.
np_valid = np.array(valid_frames)
def obtain_which_leg(cleaned_pred_indices, keypoints, task, runner_direction):
    pred_to_leg_dict = {}
    for p in cleaned_pred_indices:
        pose_kpts = keypoints[p]
        left_ankle = pose_kpts[15]
        right_ankle = pose_kpts[16]
        if runner_direction == "right":
            if task == "D":
                if right_ankle[0] > left_ankle[0]:
                    pred_to_leg_dict[p] = "R"
                else:
                    pred_to_leg_dict[p] = "L"
            elif task == "U":
                if right_ankle[0] > left_ankle[0]:
                    pred_to_leg_dict[p] = "L"
                else:
                    pred_to_leg_dict[p] = "R"

        elif runner_direction == "left":
            if task == "D":
                if right_ankle[0] > left_ankle[0]:
                    pred_to_leg_dict[p] = "L"
                else:
                    pred_to_leg_dict[p] = "R"
            elif task == "U":
                if right_ankle[0] > left_ankle[0]:
                    pred_to_leg_dict[p] = "R"
                else:
                    pred_to_leg_dict[p] = "L"
    return pred_to_leg_dict

kpts_reshaped = torch.tensor(flattened_kpts).reshape([len(valid_frames), 21, 2]).tolist()

d_frame_preds_legs_indices = obtain_which_leg(cleaned_d_preds_indices, kpts_reshaped, "D", "right")
u_frame_preds_legs_indices = obtain_which_leg(cleaned_u_preds_indices, kpts_reshaped, "U", "right")

print(len(d_frame_preds_legs_indices))
print(len(u_frame_preds_legs_indices))

cap = cv2.VideoCapture(test_video_dir)

labels_path = "/Users/abhinavarora/Desktop/CadenceCV/video18_gait_labels.json"
def load_ground_truth(path=labels_path):
    #the manual labels from test.py: {"L": {frame: "d"/"u"}, "R": {frame: "d"/"u"}}. Split into d-events and
    #u-events, each as {real_frame: leg}, so i can drop the TRUE events onto the matching probability panel.
    gt_d, gt_u = {}, {}
    if os.path.isfile(path):
        raw = json.load(open(path))
        for leg in ("L", "R"):
            for f, ev in raw.get(leg, {}).items():
                (gt_d if ev == "d" else gt_u)[int(f)] = leg
    return gt_d, gt_u


def _draw_truth(ax, truth, x, pos):
    #a dot at each hand-labelled event, coloured by leg (blue=L, orange=R). the dot sits ON the probability
    #curve at that frame, so height = the model's confidence THERE: high dot = hit, dot near 0 = a miss.
    #frames with no YOLO detection arent in x, so they fall back to y=0 (a total miss, pinned to the bottom).
    leg_colour = {"L": "blue", "R": "orange"}
    frame_to_p = {int(f): float(p) for f, p in zip(np.asarray(x).tolist(), np.asarray(pos).tolist())}
    seen = set()
    for f, leg in (truth or {}).items():
        y = frame_to_p.get(int(f), 0.0)
        ax.scatter(f, y, color=leg_colour.get(leg, "gray"), s=45, zorder=4,
                   edgecolor="k", linewidth=0.5,
                   label=(f"true {leg}" if leg not in seen else None))
        seen.add(leg)


def plot_event_probabilities(frames, d_probs, u_probs, d_events=None, u_events=None,
                             gt_d=None, gt_u=None, d_thresh=0.5, u_thresh=0.43):
    #Per-frame positive-class probability for BOTH event models over the real video frames. This is the
    #"whats actually happening" view: clean isolated peaks that cross the threshold = healthy detection;
    #flat/jittery stretches, or peaks that never cross, = the model struggling on those frames. The kept
    #events (what survived clean_preds) are dotted on so i can see which peak each cluster collapsed to.
    #The dotted vertical lines are the MANUAL ground truth - lining them up against the peaks shows hits,
    #misses, and any timing offset between what i labelled and where the model actually fires.
    x = np.asarray(frames)
    d_pos = np.asarray(d_probs)[:, 1]
    u_pos = np.asarray(u_probs)[:, 1]

    #two stacked panels sharing the frame axis - d and u have different thresholds so overlaying them lies
    fig, (ax_d, ax_u) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    ax_d.plot(x, d_pos, color="tab:red", lw=1, label="P(strikefoot)")
    ax_d.axhline(d_thresh, color="k", ls="--", lw=0.8, alpha=0.5, label=f"threshold {d_thresh}")
    _draw_truth(ax_d, gt_d, x, d_pos)   #actual d (contact) frames belong on the strikefoot panel
    if d_events:
        ax_d.scatter(x[d_events], d_pos[d_events], color="tab:red", s=40, zorder=3, label="kept event")
    ax_d.set_ylabel("P(event)"); ax_d.set_ylim(-0.02, 1.02)
    ax_d.set_title("d-frame model  (strikefoot / foot contact)"); ax_d.legend(loc="upper right")

    ax_u.plot(x, u_pos, color="tab:blue", lw=1, label="P(toe-off)")
    ax_u.axhline(u_thresh, color="k", ls="--", lw=0.8, alpha=0.5, label=f"threshold {u_thresh}")
    _draw_truth(ax_u, gt_u, x, u_pos)   #actual u (toe-off) frames belong on the toe-off panel
    if u_events:
        ax_u.scatter(x[u_events], u_pos[u_events], color="tab:blue", s=40, zorder=3, label="kept event")
    ax_u.set_ylabel("P(event)"); ax_u.set_xlabel("video frame"); ax_u.set_ylim(-0.02, 1.02)
    ax_u.set_title("u-frame model  (toe-off)"); ax_u.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #see the raw per-frame probabilities (with the manual ground-truth events overlaid) first,
    #THEN step through the detected event frames
    gt_d, gt_u = load_ground_truth()
    plot_event_probabilities(valid_frames, d_frame_probablities, u_frame_probabilities,
                             cleaned_d_preds_indices, cleaned_u_preds_indices, gt_d, gt_u)
    visualise_frames(test_video_dir, valid_frames, cleaned_d_preds_indices, non_normal_kpts)

#Pattern: L, L, R, L, L, L, R