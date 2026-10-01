#Same as archive/extract_normalised_data.py but the keypoints go through the side correction FIRST, so the LSTM is
#trained on the same kind of keypoints it gets at inference (inference runs correct_side before building features).
#Writes NEW files next to the old ones instead of overwriting them, so the uncorrected data is still there to compare
import json
import sys
import numpy as np
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src/inference")
from utils.obtain_metrics import compute_all_metrics
from utils.numpy_encoder import NumpyEncoder
from side_correction import correct_side_auto, FLIP_IDX

data_dir = "/Users/abhinavarora/Desktop/CadenceCV/ml/data"

with open(f"{data_dir}/frame_by_frame_keypoint_data.json", "r") as file:
    data = json.load(file)

#Grouping every frame entry by video (extension stripped off the name)
v_dict = {}
for x in data:
    v = x["video"].split(".")[0]
    v_dict.setdefault(v, []).append(x)

with open(f"{data_dir}/strikefoot_data.json", "r") as file:
    strikefoot_data = json.load(file)


def video_features(kpts, real_frames, v):
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
        "frames": list(range(len(kpts))),
    }
    results = compute_all_metrics(coords=coords)
    for r in results:
        #Mapping the positional index back to the real frame number so this stays joinable with strikefoot_data
        r["frame"] = real_frames[r["frame"]]
        r["video"] = v
    return results


def labelled_foot_in_front(results, strikes):
    #at a strike the striking foot is the one IN FRONT, and ankle_x_dist > 0 means "this foot is ahead in the
    #direction of travel". So this counts how many labelled strikes have the labelled leg in front
    side_map = {"L": "left", "R": "right"}
    lookup = {(r["frame"], r["side"]): r["ankle_x_dist"] for r in results}
    return sum(lookup.get((s["frame"], side_map[s["side"]]), 0) > 0 for s in strikes)


all_results = []
for v, entries in v_dict.items():
    #Sorting by frame so the velocities (np.gradient) are computed on sequential frames, and keeping the
    #real frame numbers so we can map back to them at the end
    entries = sorted(entries, key=lambda e: e["Frame"])
    real_frames = [e["Frame"] for e in entries]

    #The only new step: undo YOLO's left/right flicker before any feature is computed
    kpts, swapped = correct_side_auto([e["Keypoints"] for e in entries])
    results = video_features(kpts, real_frames, v)

    #correct_side_auto names the legs by YOLO's majority, but on clips where YOLO flickers 30-40% of the time thats
    #basically a coin flip, and a whole video named the wrong way round = every label on the wrong leg. At inference
    #the user's anchor click picks the naming, so here the LABELS play that role: keep whichever naming puts the
    #labelled foot in front at more of its strikes. This only ever flips the WHOLE video, never single frames
    strikes = [s for s in strikefoot_data if s["video"] == v]
    flipped_results = video_features(kpts[:, FLIP_IDX, :], real_frames, v)
    in_front, in_front_flipped = labelled_foot_in_front(results, strikes), labelled_foot_in_front(flipped_results, strikes)
    if in_front_flipped > in_front:
        results = flipped_results
    print(f"{v}: flipped {swapped.sum()} of {len(entries)} frames | labelled foot in front {max(in_front, in_front_flipped)}/{len(strikes)}"
          f"{' (whole video renamed)' if in_front_flipped > in_front else ''}")

    all_results.extend(results)

with open(f"{data_dir}/normalised_frame_by_frame_data_side_corrected.json", "w") as file:
    json.dump(all_results, file, indent=4, cls=NumpyEncoder)

print(f"Wrote {len(all_results)} rows across {len(v_dict)} videos")


#Same lookup + merge onto the strikes as the original script. The strike labels themselves (frame, side, u_frame)
#dont change, only the features attached to them
feature_keys = [k for k in all_results[0] if k not in ("video", "frame", "side")]
lookup = {(r["video"], r["frame"], r["side"]): r for r in all_results}

#strikefoot_data uses L/R, the normalised frame data uses left/right
side_map = {"L": "left", "R": "right"}

normalised_strikefoot = []
missed = 0
for strike in strikefoot_data:
    key = (strike["video"], strike["frame"], side_map.get(strike["side"]))
    match = lookup.get(key)
    #No normalised row for this strike means its frame wasnt in the keypoint data, so skip it
    if match is None:
        missed += 1
        continue
    new_row = dict(strike)
    for k in feature_keys:
        new_row[k] = match[k]
    normalised_strikefoot.append(new_row)

with open(f"{data_dir}/normalised_strikefoot_data_side_corrected.json", "w") as file:
    json.dump(normalised_strikefoot, file, indent=4, cls=NumpyEncoder)

print(f"Wrote {len(normalised_strikefoot)} strikes ({missed} had no keypoint match)")
