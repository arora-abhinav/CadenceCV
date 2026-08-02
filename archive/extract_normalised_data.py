import json
import sys
import numpy as np
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from utils.obtain_metrics import compute_all_metrics
from utils.numpy_encoder import NumpyEncoder

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/frame_by_frame_keypoint_data.json", "r") as file:
    data = json.load(file)

#Grouping every frame entry by video (extension stripped off the name)
v_dict = {}
for x in data:
    v = x["video"].split(".")[0]
    v_dict.setdefault(v, []).append(x)

all_results = []

for v, entries in v_dict.items():
    #Sorting by frame so the velocities (np.gradient) are computed on sequential frames, and keeping the
    #real frame numbers so we can map back to them at the end
    entries = sorted(entries, key=lambda e: e["Frame"])
    real_frames = [e["Frame"] for e in entries]

    #compute_all_metrics does numpy indexing (arr[:, 0]) so each coord list has to be an (N, 2) array.
    #Keypoint indices follow the COCO layout (5/6 shoulders, 11-16 hips/knees/ankles)
    coords = {
        "left_ankle_coords": np.array([e["Keypoints"][15] for e in entries]),
        "right_ankle_coords": np.array([e["Keypoints"][16] for e in entries]),
        "left_hip_coords": np.array([e["Keypoints"][11] for e in entries]),
        "right_hip_coords": np.array([e["Keypoints"][12] for e in entries]),
        "left_knee_coords": np.array([e["Keypoints"][13] for e in entries]),
        "right_knee_coords": np.array([e["Keypoints"][14] for e in entries]),
        "left_shoulder_coords": np.array([e["Keypoints"][5] for e in entries]),
        "right_shoulder_coords": np.array([e["Keypoints"][6] for e in entries]),
        #Positional indices 0..N-1 - compute_all_metrics uses these to index the arrays above
        "frames": list(range(len(entries))),
    }

    #Returns a list of per-frame dicts (one per side per frame) with 'frame' set to the positional index
    results = compute_all_metrics(coords=coords)

    for r in results:
        #Mapping the positional index back to the real frame number so this stays joinable with strikefoot_data
        r["frame"] = real_frames[r["frame"]]
        r["video"] = v

    all_results.extend(results)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/normalised_frame_by_frame_data.json", "w") as file:
    json.dump(all_results, file, indent=4, cls=NumpyEncoder)

print(f"Wrote {len(all_results)} rows across {len(v_dict)} videos")


#Now the same for strikefoot_data. A strike's velocities cant be computed from the single strike frame, so
#instead of recomputing we just look up the normalised features we already built for every frame and merge
#them onto each strike, keeping the strike-only fields (u_frame, strike_pattern, overstriding)
feature_keys = [k for k in all_results[0] if k not in ("video", "frame", "side")]
lookup = {(r["video"], r["frame"], r["side"]): r for r in all_results}

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/strikefoot_data.json", "r") as file:
    strikefoot_data = json.load(file)

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
    #Overwriting the old raw-pixel features with the normalised ones, keeping everything else
    new_row = dict(strike)
    for k in feature_keys:
        new_row[k] = match[k]
    normalised_strikefoot.append(new_row)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/normalised_strikefoot_data.json", "w") as file:
    json.dump(normalised_strikefoot, file, indent=4, cls=NumpyEncoder)

print(f"Wrote {len(normalised_strikefoot)} strikes to normalised_strikefoot_data.json ({missed} had no keypoint match)")
