# Multi-feature continuity (exchange) test: instead of pairing the two feet frame-to-frame on ANKLE POSITION
# alone, i stack several temporal features into each foot's STATE - ankle (x,y), signed foot angle, and knee
# flexion - each standardised so no feature dominates, then pair by whichever assignment keeps the whole
# state continuous (swap < keep vs a constant-velocity prediction). The idea: each feature covers the others'
# blind spots (foot angle for the big stance/swing contrast, knee for robustness, ankle-y for crossings).
#
# Measures, vs position-only: residual ankle-x crossings (target ~34, position-only gave 39) and the
# strike-frame leg accuracy against the manual Video18 labels.
import json
import pickle
import numpy as np

KPTS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl"
LABELS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/video18_gait_labels.json"
#slot-0 = YOLO "left" (kpts 15/17/19 + 11/13), slot-1 = YOLO "right" (16/18/20 + 12/14)


def sign_flips(diff):
    s = np.sign(diff); s[s == 0] = 1
    return int((s[:-1] != s[1:]).sum())


def signed_angle(vec, direction):
    return np.degrees(np.arctan2(vec[:, 1], direction * vec[:, 0]))


def knee_flexion(hip, knee, ankle):
    v1, v2 = hip - knee, ankle - knee
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def continuity(s0, s1, anchor, left_is_det0):
    #s0,s1: (n, D) per-frame feature STATE for slot-0 / slot-1. pair to two tracks A,B by min combined
    #displacement vs a constant-velocity prediction. returns swapped[t] (True where track A took slot-1).
    n, D = s0.shape
    A = np.zeros((n, D)); B = np.zeros((n, D)); swapped = np.zeros(n, bool)
    if left_is_det0:
        A[anchor], B[anchor] = s0[anchor], s1[anchor]
    else:
        A[anchor], B[anchor] = s1[anchor], s0[anchor]; swapped[anchor] = True

    def step(t, predA, predB):
        keep = np.linalg.norm(predA - s0[t]) + np.linalg.norm(predB - s1[t])
        swap = np.linalg.norm(predA - s1[t]) + np.linalg.norm(predB - s0[t])
        if swap < keep:
            A[t], B[t], swapped[t] = s1[t], s0[t], True
        else:
            A[t], B[t] = s0[t], s1[t]

    for t in range(anchor + 1, n):
        step(t, 2*A[t-1]-A[t-2] if t-2 >= anchor else A[t-1], 2*B[t-1]-B[t-2] if t-2 >= anchor else B[t-1])
    for t in range(anchor - 1, -1, -1):
        step(t, 2*A[t+1]-A[t+2] if t+2 < n else A[t+1], 2*B[t+1]-B[t+2] if t+2 < n else B[t+1])
    return swapped


def main():
    r = pickle.load(open(KPTS_PATH, "rb"))
    vf = np.array(r["Valid Frames"])
    K = np.array(r["Normalised Keypoints"]).reshape(len(vf), 21, 2)
    direction = 1 if np.median((K[:, 5, 0]+K[:, 6, 0])/2 - (K[:, 11, 0]+K[:, 12, 0])/2) > 0 else -1

    #per-slot raw features
    ax0, ay0 = K[:, 15, 0], K[:, 15, 1]            #slot-0 ankle
    ax1, ay1 = K[:, 16, 0], K[:, 16, 1]            #slot-1 ankle
    fa0 = signed_angle(K[:, 17, :] - K[:, 19, :], direction)   #slot-0 foot angle
    fa1 = signed_angle(K[:, 18, :] - K[:, 20, :], direction)
    kf0 = knee_flexion(K[:, 11, :], K[:, 13, :], K[:, 15, :])  #slot-0 knee flexion
    kf1 = knee_flexion(K[:, 12, :], K[:, 14, :], K[:, 16, :])

    #standardise each feature by its pooled std (both slots) so distances are comparable
    def std_pair(a, b):
        s = np.concatenate([a, b]).std() + 1e-9
        return a / s, b / s
    zx0, zx1 = std_pair(ax0, ax1); zy0, zy1 = std_pair(ay0, ay1)
    zf0, zf1 = std_pair(fa0, fa1); zk0, zk1 = std_pair(kf0, kf1)

    #state sets: position-only (ankle x,y) vs ensemble (ankle x,y + foot angle + knee flexion)
    pos0 = np.stack([zx0, zy0], 1);            pos1 = np.stack([zx1, zy1], 1)
    ens0 = np.stack([zx0, zy0, zf0, zk0], 1);  ens1 = np.stack([zx1, zy1, zf1, zk1], 1)

    #anchor = first max-separation frame of the ankles in x
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.abs(ax0 - ax1))
    anchor = int(peaks[0]) if len(peaks) else int(np.argmax(np.abs(ax0-ax1)[1:-1])) + 1

    #ground-truth strikes
    lab = json.load(open(LABELS_PATH)); f2i = {int(f): i for i, f in enumerate(vf)}
    strikes = [(int(f), leg) for leg in ("L", "R") for f, v in lab[leg].items() if v == "d" and int(f) in f2i]

    def evaluate(s0, s1, name):
        for lid in (True, False):
            sw = continuity(s0, s1, anchor, lid)
            #resulting tracks' ankle-x (trackA takes slot-1 where swapped) -> crossings
            trackA_x = np.where(sw, ax1, ax0); trackB_x = np.where(sw, ax0, ax1)
            crossings = sign_flips(trackA_x - trackB_x)
            #strike leg accuracy: striking foot = the LOW foot-angle track; name it via the anchor (A=left)
            hit = 0
            for f, leg in strikes:
                i = f2i[f]
                a_ang, b_ang = (fa1[i], fa0[i]) if sw[i] else (fa0[i], fa1[i])
                pred = "L" if a_ang < b_ang else "R"      # A = left
                hit += (pred == leg)
            print(f"  {name:14s} left_is_det0={str(lid):5s}: ankle-x crossings {crossings:3d}  "
                  f"strike acc {hit}/{len(strikes)} = {hit/len(strikes):.0%}")

    print(f"anchor frame {vf[anchor]} | direction {'right' if direction>0 else 'left'} | strikes {len(strikes)}")
    print("reference: raw position-only continuity earlier = 39 crossings, 59% strike acc; alternation = 100%")
    evaluate(pos0, pos1, "position-only")
    evaluate(ens0, ens1, "ensemble(4D)")


if __name__ == "__main__":
    main()
