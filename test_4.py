# Per-frame left/right keypoint correction via knee velocity + inter-knee distance, seeded by a user-marked
# two-frame anchor. Independent of the SVM events, so it works even when strike detection misses frames.
#
# Heuristic: d(t)=|xL-xR| (inter-knee distance) is SWITCH-INVARIANT, so it and its derivative dd are reliable.
# Each knee's x-velocity direction should only reverse at a distance EXTREMUM. If BOTH knees reverse direction
# at a frame where dd is clearly non-zero (mid approach/separation, not an extremum), thats non-physical -> a
# YOLO label switch. The anchor (two consecutive frames the user marks) pins velocity direction + which knee
# is truly left, and fixes the absolute orientation of the whole corrected sequence.
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

KIN_OUT = "/Users/abhinavarora/Desktop/CadenceCV/knee_kinematics.json"

KPTS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl"
TEST_VIDEO = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4"
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
L_KNEE, R_KNEE = 13, 14
VEL_FLOOR = 1e-3
_NEXT_KEYS = {65363, 2555904, 63235}


def detect_switches(xL, xR):
    #returns the frames where YOLO switched the two knees, per the velocity + distance heuristic
    d = np.abs(xL - xR)                       # switch-invariant inter-knee distance
    dd = np.gradient(d)                       # <0 approaching, >0 separating, ~0 at an extremum
    vL, vR = np.gradient(xL), np.gradient(xR) # per-knee x-velocity
    dd_thresh = np.percentile(np.abs(dd), 35) # 'clearly non-zero' -> away from an extremum

    def vsign(v):
        return 0 if abs(v) < VEL_FLOOR else (1 if v > 0 else -1)

    switches = []
    for t in range(1, len(xL)):
        # ================== MY HEURISTIC (inter-knee distance <-> knee velocities) ==================
        # A knee's x-velocity is only ALLOWED to reverse direction at a distance extremum. Two cases:
        #   - dd < 0 (knees APPROACHING): each knee holds its direction until they meet at a crossing.
        #     If both suddenly reverse WHILE still approaching -> not physical -> a switch.
        #   - dd > 0 (knees SEPARATING): likewise the directions must stay consistent with separating.
        # So a legitimate reversal must coincide with dd ~ 0 (an extremum). A reversal of BOTH knees at
        # once, at a frame where dd is clearly non-zero (mid approach/separation), is a YOLO leg switch.
        lflip = vsign(vL[t]) and vsign(vL[t-1]) and vsign(vL[t]) != vsign(vL[t-1])   # left knee reversed dir?
        rflip = vsign(vR[t]) and vsign(vR[t-1]) and vsign(vR[t]) != vsign(vR[t-1])   # right knee reversed dir?
        if lflip and rflip and abs(dd[t]) > dd_thresh:   # both reversed AND not at a distance extremum
            switches.append(t)
        # ============================================================================================
    return switches, d, dd


def detect_switches_bidirectional(xL, xR):
    #Run the SAME heuristic forward AND on the time-reversed keypoints. Reversing time flips the sign of the
    #distance derivative (a decreasing inter-knee distance becomes increasing) and the knee velocities, so its
    #the mirror-image of the heuristic. A REAL switch is a sharp single-frame teleport, so both directions
    #localise it to the same frame -> they AGREE. A crossing/jitter false-positive gets smeared to different
    #frames forward vs backward -> they DISAGREE. So i only trust a switch where the two agree.
    n = len(xL)
    fwd, d, dd = detect_switches(xL, xR)
    bwd_rev, _, _ = detect_switches(xL[::-1], xR[::-1])
    #map a reversed-time index k back to its forward frame: reversed (k-1->k) is original (n-k -> n-1-k),
    #which the forward pass would label as frame n-k
    bwd = {n - k for k in bwd_rev}
    fwd = set(fwd)
    confirmed = sorted(fwd & bwd)          #both directions flag the same frame -> a trustworthy switch
    disagreements = sorted(fwd ^ bwd)      #only ONE direction flagged it -> untrusted (crossing/jitter)
    return confirmed, disagreements, d, dd


def mark_anchor_cv2(video_path, vf, px, d):
    #visible anchor: cycle the most-separated frames (clear left/right), each showing frame a AND a+1 side by
    #side with the two knee dots. 1 = blue(kpt13) is LEFT, 2 = orange(kpt14) is LEFT, ->/n next, q quit.
    peaks, _ = find_peaks(d)
    peaks = sorted(peaks.tolist(), key=lambda p: -d[p]) or [int(np.argmax(d[1:-1])) + 1]
    cap = cv2.VideoCapture(video_path)
    i, result = 0, None
    while True:
        a = peaks[i]
        panels = []
        for f in (a, a + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf[f]))
            ret, img = cap.read()
            if not ret:
                img = np.zeros((400, 300, 3), np.uint8)
            for lbl, idx, col in [("1", L_KNEE, (255, 0, 0)), ("2", R_KNEE, (0, 140, 255))]:
                x, y = int(px[f, idx, 0]), int(px[f, idx, 1])
                cv2.circle(img, (x, y), 9, col, -1)
                cv2.putText(img, lbl, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
            cv2.putText(img, f"frame {int(vf[f])}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panels.append(img)
        hgt = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (int(p.shape[1] * hgt / p.shape[0]), hgt)) for p in panels]
        combined = np.hstack(panels)
        cv2.putText(combined, "LEFT knee?  1=blue  2=orange    ->/n next    q quit",
                    (15, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        H, W = combined.shape[:2]
        if H > 900:
            combined = cv2.resize(combined, (int(W * 900 / H), 900))
        cv2.imshow("anchor: mark the LEFT knee across two consecutive frames", combined)
        key = cv2.waitKeyEx(0); low = key & 0xFF
        if key in _NEXT_KEYS or low in (ord("n"), ord(".")):
            i = (i + 1) % len(peaks)
        elif low == ord("1"):
            result = (a, True); break
        elif low == ord("2"):
            result = (a, False); break
        elif low in (ord("q"), 27):
            result = None; break
    cap.release(); cv2.destroyAllWindows()
    return result


def visualise_corrected_keypoints(video_path, vf, corrected_px, flipped):
    #step through EVERY frame showing the CORRECTED pixel keypoints, so i can eyeball whether the legs are
    #right after the flip. knees/ankles/feet get named labels + colours so left vs right is obvious.
    #space/n = next, b = back, q = quit. banner says whether THIS frame was flipped by the heuristic.
    knee_names = {13: "L knee", 14: "R knee"}
    ankle_names = {15: "L ankle", 16: "R ankle"}
    foot_names = {17: "L toe", 18: "R toe", 19: "L heel", 20: "R heel"}
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf[i]))
        ret, img = cap.read()
        if not ret:
            i = min(i + 1, len(vf) - 1); continue
        pose = corrected_px[i]
        for idx, (x, y) in enumerate(pose):
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)          #every keypoint a small green dot
            if idx in knee_names:                                #knees orange
                cv2.circle(img, (x, y), 6, (0, 165, 255), -1)
                cv2.putText(img, knee_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 1)
            elif idx in ankle_names:                             #ankles yellow
                cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(img, ankle_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            elif idx in foot_names:                              #feet red
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, foot_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        tag = "FLIPPED" if flipped[i] else "as-is"
        cv2.putText(img, f"frame {int(vf[i])}  ({tag})   space/n next  b back  q quit",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        H, W = img.shape[:2]
        if H > 900:
            img = cv2.resize(img, (int(W * 900 / H), 900))
        cv2.imshow("corrected keypoints per frame", img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("n"), ord(" ")):
            i = min(i + 1, len(vf) - 1)
        elif key == ord("b"):
            i = max(i - 1, 0)
        elif key in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows()


def build_flip_state(n, switches, anchor, anchor_swapped):
    #relative toggle pattern from the switch frames; absolute orientation pinned by the anchor mark
    rel = np.zeros(n, bool)
    cur, sw = False, set(switches)
    for t in range(n):
        if t in sw:
            cur = not cur
        rel[t] = cur
    if rel[anchor] != anchor_swapped:
        rel = ~rel
    return rel


def main():
    r = pickle.load(open(KPTS_PATH, "rb"))
    vf = np.array(r["Valid Frames"])
    K = np.array(r["Normalised Keypoints"]).reshape(len(vf), 21, 2)
    px = np.array(r["Non-Normalised Keypoints"])
    n = len(vf)

    xL, xR = K[:, L_KNEE, 0], K[:, R_KNEE, 0]
    #your velocity + dd heuristic (forward + backward with agreement) - the only detector now
    switches, disagreements, d, dd = detect_switches_bidirectional(xL, xR)
    print(f"[velocity+dd] switches: {switches}")
    print(f"[velocity+dd] disagreements (only one direction): {disagreements}")

    #save the raw kinematics the heuristic runs on: inter-knee distance + each leg's knee x-velocity, per frame
    vL, vR = np.gradient(xL), np.gradient(xR)
    json.dump({
        "frames": vf.tolist(),
        "inter_knee_distance": d.tolist(),
        "left_knee_velocity": vL.tolist(),
        "right_knee_velocity": vR.tolist(),
    }, open(KIN_OUT, "w"), indent=2)
    print(f"saved knee kinematics -> {KIN_OUT}")

    marked = mark_anchor_cv2(TEST_VIDEO, vf, px, d)
    if marked is None:
        print("marking aborted"); return
    anchor, left_is_kpt13 = marked
    flipped = build_flip_state(n, switches, anchor, anchor_swapped=not left_is_kpt13)

    #apply the whole-body flip on flagged frames -> corrected keypoints (normalised AND pixel copies)
    corrected = K.copy()
    corrected[flipped] = corrected[flipped][:, FLIP_IDX, :]
    corrected_px = px.copy()
    corrected_px[flipped] = corrected_px[flipped][:, FLIP_IDX, :]
    print(f"anchor ({anchor},{anchor+1}) | left = {'kpt13' if left_is_kpt13 else 'kpt14'} | "
          f"flipped {int(flipped.sum())}/{n} frames")

    #visualise: raw vs corrected knee-x tracks, with the anchor + detected switches marked
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax0.plot(vf, xL, color="tab:blue", lw=1.1, label="kpt13 (YOLO left knee)")
    ax0.plot(vf, xR, color="tab:orange", lw=1.1, label="kpt14 (YOLO right knee)")
    for s in switches:
        ax0.axvline(vf[s], color="red", ls=":", lw=0.8)
    ax0.axvline(vf[anchor], color="green", lw=1.5, label="anchor")
    ax0.set_ylabel("knee x (raw)"); ax0.legend(loc="upper right")
    ax0.set_title(f"raw knee-x (red = detected switches, green = anchor)  |  {len(switches)} switches")

    cL, cR = corrected[:, L_KNEE, 0], corrected[:, R_KNEE, 0]
    ax1.plot(vf, cL, color="tab:blue", lw=1.1, label="corrected LEFT knee")
    ax1.plot(vf, cR, color="tab:orange", lw=1.1, label="corrected RIGHT knee")
    ax1.axvline(vf[anchor], color="green", lw=1.5)
    ax1.set_ylabel("knee x (corrected)"); ax1.set_xlabel("frame"); ax1.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    #step through Video18 frame by frame with the corrected keypoints drawn on
    visualise_corrected_keypoints(TEST_VIDEO, vf, corrected_px, flipped)


if __name__ == "__main__":
    main()
