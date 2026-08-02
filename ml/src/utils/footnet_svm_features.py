#Bolt the FootNet LSTM's per-frame kinematic features onto the SVM feature vectors. The SVM otherwise sees
#only a single frame's 42 flattened keypoint coords (21 keypoints x 2); these add the TEMPORAL signal
#(velocities) the raw pose cant carry: per leg -> ankle_x_vel, tibial_angle, shin_velocity, ankle_y_vel,
#inter-ankle x-distance = 5 features x 2 legs = 10 columns.
#
#NOTE ON DIRECTION: unlike the LSTM's compute_all_metrics, these are NOT canonicalised to rightward travel.
#Direction is kept CONSTANT (as-is), so the features vary with which way the runner faces - matching the raw
#keypoints, which arent canonicalised either. Direction-invariance is left to the mirror AUGMENTATION.
#
#Gotchas:
# - velocity is np.gradient over the per-video sequence, so i rebuild each video's ordered coord arrays from
#   the full frame-by-frame json (a sampled training row has no neighbours to differentiate against).
# - YOLO drops undetected frames, so per-video frames are NON-contiguous; gradient treats array-adjacent
#   rows as 1 apart even when theyre really several frames apart. Same approximation the LSTM lived with.
# - because the features are direction-VARYING now, the training mirror cant just swap L/R - it needs the
#   features of the actually-mirrored sequence, so i precompute those too (the "mirror" entry).
import json
import os
from collections import defaultdict

import numpy as np

FOOTNET_KEYS = ["ankle_x_vel", "tibial_angle", "shin_velocity", "ankle_y_vel", "ankle_x_dist"]
#10 columns: both legs, L block then R block
FOOTNET_COLS = [f"{side}_{k}" for side in ("L", "R") for k in FOOTNET_KEYS]

#YOLO/COCO indices for the joints the features need
L_ANKLE, R_ANKLE, L_KNEE, R_KNEE = 15, 16, 13, 14
#left<->right keypoint pairing for the horizontal mirror (same list the model files use for augmentation)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]

FRAME_BY_FRAME_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/frame_by_frame_keypoint_data.json"


def base_name(v):
    #frame data stores 'Video8.mp4' but some callers strip the extension - normalise both to 'Video8'
    return v.split(".")[0]


def _five_features(kpts):
    #kpts: (n, 21, 2) NORMALISED keypoints in playback order. Returns (n, 10) in FOOTNET_COLS order
    #(L block then R block). FIXED direction - no canonicalisation, everything left as-is.
    kpts = np.asarray(kpts, dtype=float)
    n = len(kpts)
    out = np.zeros((n, len(FOOTNET_COLS)), dtype=float)
    if n < 2:
        return out  #np.gradient needs at least 2 samples

    def fill(ankle_i, knee_i, other_ankle_i, base):
        ankle = kpts[:, ankle_i, :]
        shin = kpts[:, knee_i, :] - ankle           #knee -> ankle vector (shin bone)
        other_ankle_x = kpts[:, other_ankle_i, 0]
        out[:, base + 0] = np.gradient(ankle[:, 0])                       #ankle_x_vel
        out[:, base + 1] = np.pi / 2 - np.arctan2(shin[:, 1], shin[:, 0]) #tibial_angle
        out[:, base + 2] = np.gradient(shin[:, 0])                        #shin_velocity
        out[:, base + 3] = np.gradient(ankle[:, 1])                       #ankle_y_vel
        out[:, base + 4] = ankle[:, 0] - other_ankle_x                    #inter-ankle x distance (signed)

    fill(L_ANKLE, L_KNEE, R_ANKLE, 0)                #left leg -> cols 0..4
    fill(R_ANKLE, R_KNEE, L_ANKLE, len(FOOTNET_KEYS))#right leg -> cols 5..9
    return out


def _mirror_sequence(kpts):
    #horizontal flip: x -> 1 - x, and swap the left/right keypoint slots via FLIP_IDX. Same transform the
    #model files apply to the keypoints for augmentation, so the mirrored FEATURES stay consistent with them.
    kpts = np.asarray(kpts, dtype=float)
    m = np.empty_like(kpts)
    for i in range(21):
        src = FLIP_IDX[i]
        m[:, i, 0] = 1.0 - kpts[:, src, 0]
        m[:, i, 1] = kpts[:, src, 1]
    return m


def build_footnet_lookup(frame_by_frame_path=FRAME_BY_FRAME_PATH):
    #-> {(video_base, real_frame): {"orig": (10,), "mirror": (10,)}}. Built ONCE from every frame so the
    #gradients have their neighbours. "mirror" = the features of the flipped sequence, for the augmentation.
    data = json.load(open(frame_by_frame_path))
    per_video = defaultdict(list)
    for r in data:
        per_video[base_name(r["video"])].append((r["Frame"], r["Keypoints"]))

    lookup = {}
    for video, rows in per_video.items():
        rows.sort(key=lambda t: t[0])
        real_frames = [t[0] for t in rows]
        kpts = np.asarray([t[1] for t in rows], dtype=float)  # (n, 21, 2)
        orig = _five_features(kpts)
        mirror = _five_features(_mirror_sequence(kpts))
        for pos, rf in enumerate(real_frames):
            lookup[(video, rf)] = {"orig": orig[pos], "mirror": mirror[pos]}
    return lookup


def attach_footnet_features(df, lookup):
    #add the 10 feature columns to an SVM dataframe, matched on (video, Frame). Missing lookups -> 0.0.
    df = df.copy()
    for col in FOOTNET_COLS:
        df[col] = 0.0
    for i in df.index:
        entry = lookup.get((base_name(df.at[i, "video"]), df.at[i, "Frame"]))
        if entry is None:
            continue
        for col, val in zip(FOOTNET_COLS, entry["orig"]):
            df.at[i, col] = val
    return df


def mirror_footnet_features(df, lookup):
    #the augmentation flips the keypoints; here we swap in the features of the MIRRORED sequence so pose and
    #features stay consistent. NOT a plain L/R swap anymore - these features are direction-varying, so a flip
    #changes velocities/angles, not just which leg is which. Matched by (video, Frame) like attach.
    df = df.copy()
    for i in df.index:
        entry = lookup.get((base_name(df.at[i, "video"]), df.at[i, "Frame"]))
        if entry is None:
            continue
        for col, val in zip(FOOTNET_COLS, entry["mirror"]):
            df.at[i, col] = val
    return df


def footnet_features_for_sequence(kpts_seq):
    #Inference-side path: one video's NORMALISED keypoints as (n, 21, 2) in playback order -> (n, 10) in
    #FOOTNET_COLS order, ready to hstack onto the 42 flattened keypoints. Same fixed-direction maths as
    #build_footnet_lookup's "orig", so train and inference cant drift.
    return _five_features(kpts_seq)
