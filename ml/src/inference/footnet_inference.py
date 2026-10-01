import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.footnet_model import CustomDataLoader, LSTM_custom
import torch
import numpy as np
# configure_data builds the windows. compute_all_metrics is called with coords= only (normalised keypoints
# that were already extracted), so it never re-runs YOLO from here.
from utils.obtain_metrics import configure_data, compute_all_metrics
# Same YOLO -> normalised -> side corrected keypoints the SVM inference uses
from utils.normalised_keypoints import load_corrected_keypoints
import pickle
from torch.utils.data import DataLoader
import cv2

weights_dir = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights"
test_video_dir = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4"

#The threshold the LSTM was actually validated at (footnet_model.py's eval + the head to head vs the SVM).
#The old 0.65 here was never tested anywhere
contact_threshold = 0.35


#Builds the LSTM's frame_by_frame_data the EXACT same way the training data was built
#(archive/extract_normalised_data.py): bbox normalised keypoints -> compute_all_metrics with positional frames
#-> map the positional index back to the real frame number. If inference builds these any differently the LSTM
#is being fed inputs it never saw during training
def build_frame_by_frame_data(normalised_kpts, valid_frames):
    #(n, 21, 2) so the COCO indices can be pulled out per joint
    kpts = np.array(normalised_kpts).reshape(len(valid_frames), 21, 2)

    #compute_all_metrics does numpy indexing (arr[:, 0]) so each coord list has to be an (N, 2) array.
    #Keypoint indices follow the COCO layout (5/6 shoulders, 11-16 hips/knees/ankles)
    coords = {
        "left_ankle_coords": kpts[:, 15],
        "right_ankle_coords": kpts[:, 16],
        "left_hip_coords": kpts[:, 11],
        "right_hip_coords": kpts[:, 12],
        "left_knee_coords": kpts[:, 13],
        "right_knee_coords": kpts[:, 14],
        "left_shoulder_coords": kpts[:, 5],
        "right_shoulder_coords": kpts[:, 6],
        #Positional indices 0..N-1 - compute_all_metrics uses these to index the arrays above
        "frames": list(range(len(valid_frames))),
    }

    results = compute_all_metrics(coords=coords)
    for r in results:
        #Mapping the positional index back to the real frame number so it lines up with valid_frames
        r["frame"] = valid_frames[r["frame"]]

    return results


def run_inference(frame_by_frame_data):
    #Loading the model:
    model = LSTM_custom(input_size=5, hidden_size=32, num_layers=1, num_classes=1)
    model.load_state_dict(torch.load("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/footnet_lstm_best.pth", weights_only=True))
    model.eval()

    # frame_by_frame_data is already computed upstream (YOLO has already run in main.py).
    # Accepting it as a parameter means YOLO is never re-triggered by importing this file.

    #Loading mean and std from pickle file
    with open("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/scaler_means_and_dev.pkl", "rb") as file:
        data = pickle.load(file)

    means = data["Means"]
    stds = data["Stds"]

    #Running data through the CustomDataLoader
    all_features, all_masks, all_frames, all_sides = configure_data(frame_by_frame_data, means, stds)
    dataset = CustomDataLoader(all_features, all_masks)

    batch_size = 32
    #shuffle=False so batch i always maps to windows [i*batch_size, (i+1)*batch_size)
    video_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #Inference loop — storing unmasked PROBABILITIES (shape: batch_size x 40) so the
    #averaging loop can index per-window using i * batch_size
    all_predictions = []
    with torch.no_grad():
        for (ind, data) in enumerate(video_data):
            features, masks = data
            logits = model(features)
            all_predictions.append(torch.sigmoid(logits))

    #Sliding window average where each frame collects probabilities from every window that covered it
    #The predictions given by the LSTM are batched, meaning that each batch is of shape (32, 40) and not
    #(32, 40, 1) due to output.squeeze done in footnet_model. This corresponds to 32 chunks of size 40 predictions (The resampling num)
    #So, each batch must be iterated over where 32 chunks of size 40 are obtained.
    #Instead of iterating over contiguous chunks, chunks in continguous batches are instead iterated over
    #Keyed by (side, frame) so a left-foot window and a right-foot window that happen to share
    #the same frame number never dump their predictions into the same vote bucket.
    frame_to_pred_dict = {}
    for (i, batch_preds) in enumerate(all_predictions):
        #The first 40 frame window always will start at i * batch_size and the last 40 frame window will
        #be found at (i + 1) * batch_size - 1
        window_start = i * batch_size
        #Last batch may not have 32 chunks
        batch_size_actual = batch_preds.shape[0]

        #Now this is iterating over the actual 40-sized framed windows
        for j in range(batch_size_actual):
            window_idx = window_start + j
            side = all_sides[window_idx]
            #Apply mask to get only real (non-padded) frames and predictions for this window
            window_mask = all_masks[window_idx]
            #applying the specific mask to a specific window in this current batch
            window_frames = all_frames[window_idx][window_mask]
            window_preds = batch_preds[j][window_mask]

            #Finally constructing the frame_to_pred_dict : {(side, frame): predictions over each window}
            for (ind, frame) in enumerate(window_frames):
                key = (side, int(frame.item()))
                if key not in frame_to_pred_dict:
                    frame_to_pred_dict[key] = []
                frame_to_pred_dict[key].append(window_preds[ind].item())

    #Averaging the probabilities first and THEN thresholding, instead of thresholding every window and taking a
    #majority vote. This is the same way the model was scored when it was validated, so the 0.35 carries over
    frame_to_prob_dict = {}
    for key in frame_to_pred_dict:
        frame_to_prob_dict[key] = float(np.mean(frame_to_pred_dict[key]))
        frame_to_pred_dict[key] = int(frame_to_prob_dict[key] > contact_threshold)

    strikefoot_frames, toe_off_frames = obtain_gait_events(frame_to_pred_dict)
    return strikefoot_frames, toe_off_frames, frame_to_pred_dict, frame_to_prob_dict


#The LSTM gives contact (1) / no contact (0) per (side, frame), so BOTH gait events fall straight out of it per leg:
#strikefoot = the foot goes from no contact -> contact (rising edge), toe off = contact -> no contact (falling edge).
#Unlike the SVMs, the leg comes for free here since every window is already per leg, so theres no need to ask the
#user which leg the first event belongs to (obtain_which_leg)
def obtain_gait_events(frame_to_pred_dict):
    strikefoot_frames = []
    toe_off_frames = []
    for side in ["L", "R"]:
        #iterating each leg's frames in sorted order so the previous-prediction state never leaks across legs
        side_frames = sorted(f for (s, f) in frame_to_pred_dict if s == side)
        preds = [frame_to_pred_dict[(side, f)] for f in side_frames]
        for i in range(1, len(preds) - 1):
            #The new state has to hold for at least 2 frames, otherwise a single flickered frame in the middle of
            #a stance would count as a toe off AND a strike. Same rule used when the LSTM was compared to the SVM
            if preds[i - 1] == 0 and preds[i] == 1 and preds[i + 1] == 1:
                strikefoot_frames.append((side, side_frames[i]))
            elif preds[i - 1] == 1 and preds[i] == 0 and preds[i + 1] == 0:
                toe_off_frames.append((side, side_frames[i]))

    #A leg can only strike once per stride, so same leg events that are way too close together are one event that
    #flickered. Earliest strike is kept (thats the real touchdown, a flicker mid stance shows up after it) and the
    #latest toe off is kept (the flicker shows up before the real push off)
    strikefoot_frames = enforce_same_leg_spacing(strikefoot_frames, keep_first=True)
    toe_off_frames = enforce_same_leg_spacing(toe_off_frames, keep_first=False)

    #Sorted by frame so downstream metrics see events in chronological order across both legs.
    strikefoot_frames.sort(key=lambda sf: sf[1])
    toe_off_frames.sort(key=lambda sf: sf[1])
    return strikefoot_frames, toe_off_frames


#Same idea as clean_preds in svm_inference.py: the tolerance is gap_frac * the median same leg gap, so it adapts to
#cadence and frame rate instead of assuming 30 fps. gap_frac=0.4 is the same default clean_preds uses. Checked on the
#training videos it barely moves F1 (0.817 -> 0.817) but cuts false events, on the held out videos it helped a bit (0.865 -> 0.875)
def enforce_same_leg_spacing(events, keep_first, gap_frac=0.4):
    spaced = []
    for side in ["L", "R"]:
        frames = sorted(f for (s, f) in events if s == side)
        #need at least 2 gaps for the median to mean anything
        if len(frames) < 3:
            spaced += [(side, f) for f in frames]
            continue

        max_gap = gap_frac * np.median(np.diff(frames))
        #Grouping events that are closer than max_gap to the previous one
        groups = [[frames[0]]]
        for f in frames[1:]:
            if f - groups[-1][-1] < max_gap:
                groups[-1].append(f)
            else:
                groups.append([f])

        spaced += [(side, g[0] if keep_first else g[-1]) for g in groups]

    return spaced


#The whole LSTM pipeline for one video: YOLO keypoints (cached) -> FootNet features -> LSTM -> gait events.
#Returns the same things svm_inference.py used to hand to inference_metrics.py, so calculate_metrics doesnt change:
#the event dicts are {real frame: "l"/"r"} which is the format obtain_which_leg produced for the SVMs
def run_video_inference(video_dir=test_video_dir):
    #Own cache per video so it never clashes with the SVM's normalised_keypoints.pkl
    video_name = os.path.splitext(os.path.basename(video_dir))[0]
    kpts_path = os.path.join(weights_dir, f"normalised_keypoints_{video_name}.pkl")
    results = load_corrected_keypoints(video_dir, kpts_path)
    valid_frames = results["Valid Frames"]

    frame_by_frame_data = build_frame_by_frame_data(results["Normalised Keypoints"], valid_frames)
    strikefoot_frames, toe_off_frames, frame_to_pred_dict, frame_to_prob_dict = run_inference(frame_by_frame_data)

    #(side, frame) tuples -> {frame: "l"/"r"}
    d_frame_preds_legs_indices = {f: side.lower() for (side, f) in strikefoot_frames}
    u_frame_preds_legs_indices = {f: side.lower() for (side, f) in toe_off_frames}

    cap = cv2.VideoCapture(video_dir)
    return d_frame_preds_legs_indices, u_frame_preds_legs_indices, cap, valid_frames, results["Non-Normalised Keypoints"]


def visualise_strike_foot_frames(video_dir, strikefoot_frames):
    cap = cv2.VideoCapture(video_dir)
    #strikefoot_frames entries are (side, frame) tuples now
    for side, f in strikefoot_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ret, img = cap.read()
        cv2.imshow(f"Strikefoot frame ({side})", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Standalone test: loads cached YOLO output produced by main.py and runs LSTM inference.
    # To regenerate the cache (or run on a new video), run main.py first.
    cache_path = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/keypoints_cache.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    frame_by_frame_data = cache["frame_by_frame_data"]

    strikefoot_frames, toe_off_frames, frame_to_pred_dict, frame_to_prob_dict = run_inference(frame_by_frame_data)
    visualise_strike_foot_frames("/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4", strikefoot_frames)
    print(f"Strikefoot frames: {strikefoot_frames}")
    print(f"Toe off frames: {toe_off_frames}")
