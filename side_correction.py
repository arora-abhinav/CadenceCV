# ============================================================================
# side_correction.py
# ----------------------------------------------------------------------------
# Fix YOLO's flickering left/right assignment across the WHOLE dataset.
#
# Per video: track the two ankle detections by continuity (ignore YOLO's L/R
# label), seeded from an anchor frame the user hand-marks. The anchor candidates
# are the local maxima of ankle x-separation (feet split, left vs right clear);
# you can cycle through them with the right-arrow (or 'n') until you find one
# that's unambiguous, then press 1/2 for the left leg. Every frame the tracker
# says YOLO was swapped, i re-index that frame's 21 keypoints with FLIP_IDX - a
# full left<->right relabel of the whole body - so everything stays on one side.
#
# Then i apply the SAME per-(video,frame) swaps to the strikefoot (d-frame) kpts.
#
# The chosen anchor frame + orientation for each video is saved to side_anchors.json
# (a record of what anchored each correction). There is NO resume cache - every run
# re-marks every video from scratch.
# ============================================================================
import json
import os
from collections import defaultdict

import numpy as np
import cv2
from scipy.signal import find_peaks

DATA_DIR = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data"
VIDEOS_DIR = "/Users/abhinavarora/Desktop/CadenceCV/Videos"
FBF_PATH = os.path.join(DATA_DIR, "frame_by_frame_keypoint_data.json")
STRIKE_PATH = os.path.join(DATA_DIR, "strikefoot_keypoints.json")
OUT_FBF = os.path.join(DATA_DIR, "frame_by_frame_keypoint_data_corrected.json")
OUT_STRIKE = os.path.join(DATA_DIR, "strikefoot_keypoints_corrected.json")
ANCHORS_PATH = os.path.join(DATA_DIR, "side_anchors.json")   #record of the anchor frame chosen per video

#full left<->right relabel of all 21 keypoints (same list the model files use for mirroring)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
L_ANKLE, R_ANKLE = 15, 16
MAX_DISPLAY_H = 900
#right-arrow keycodes vary by platform (macOS HighGUI ~63235); 'n'/'.'/'d' work everywhere
NEXT_ARROWS = {65363, 2555904, 63235}


def base_name(v):
    return v.split(".")[0]


def resolve_video(name):
    #data video names are inconsistent ('Video8.mp4', 'instavid_1', 'Video9.MOV'...) - find the real file
    stem = base_name(name)
    for cand in (name, stem + ".mp4", stem + ".MP4", stem + ".mov", stem + ".MOV"):
        p = os.path.join(VIDEOS_DIR, cand)
        if os.path.exists(p):
            return p
    return None


def _assign(d0, d1, predA, predB):
    #hand the two detections to tracks A,B whichever way lands closer to prediction -> (A_pt, B_pt, swapped)
    keep = np.linalg.norm(predA - d0) + np.linalg.norm(predB - d1)
    swap = np.linalg.norm(predA - d1) + np.linalg.norm(predB - d0)
    if swap < keep:
        return d1, d0, True
    return d0, d1, False


def find_peak_anchors(a15, a16):
    #ALL local maxima of ankle x-separation, in frame order = the anchor candidates the user cycles through.
    #find_peaks never returns the first/last sample, so none of these is the last two frames.
    dist = np.abs(a15[:, 0] - a16[:, 0])
    peaks, _ = find_peaks(dist)
    return [int(p) for p in peaks] if len(peaks) else [int(np.argmax(dist[1:-1])) + 1]


def continuity_from_anchor(det0, det1, anchor, left_is_det0):
    #seed A=left at the anchor, then propagate continuity forward and backward
    n = len(det0)
    A = np.zeros((n, 2)); B = np.zeros((n, 2)); swapped = np.zeros(n, bool)
    if left_is_det0:
        A[anchor], B[anchor] = det0[anchor], det1[anchor]
    else:
        A[anchor], B[anchor] = det1[anchor], det0[anchor]
        swapped[anchor] = True

    for t in range(anchor + 1, n):                       #forward
        if t - 2 >= anchor:
            predA, predB = 2 * A[t-1] - A[t-2], 2 * B[t-1] - B[t-2]
        else:
            predA, predB = A[t-1], B[t-1]
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)

    for t in range(anchor - 1, -1, -1):                  #backward (uses the two LATER frames if they exist)
        if t + 2 < n and t + 1 < n:
            predA, predB = 2 * A[t+1] - A[t+2], 2 * B[t+1] - B[t+2]
        else:
            predA, predB = A[t+1], B[t+1]
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)

    return swapped


def correct_sequence(kpts, swapped):
    #on every swapped frame, relabel the whole body left<->right via FLIP_IDX
    out = kpts.copy()
    out[swapped] = out[swapped][:, FLIP_IDX, :]
    return out


def mark_orientation(video_path, frames, peaks, a15, a16):
    #The stored keypoints are BBOX-normalised (relative to the person box, not the frame), and the box isnt
    #saved anywhere - so i CANT place the ankle dots on the frame accurately (that was the "wrong coords" bug).
    #BUT bbox-normalisation is monotonic in x, so the ankle with the smaller x is still the one further LEFT
    #in the image. So i mark by IMAGE SIDE: you look at the runner and tell me which side the LEFT leg is on,
    #and i map that to the correct detection. Cycle peaks with ->/n for an unambiguous frame.
    #Returns (anchor_index, left_is_det0) | "skip" | None.
    cap = cv2.VideoCapture(video_path)
    i = 0
    result = None
    while True:
        anchor = peaks[i]
        real_frame = int(frames[anchor])
        cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame)
        ret, img = cap.read()
        if not ret:
            i = (i + 1) % len(peaks)   #unreadable peak, jump to the next candidate
            continue

        H, W = img.shape[:2]
        cv2.line(img, (W // 2, 0), (W // 2, H), (0, 255, 255), 1)            #image midline for reference
        cv2.putText(img, "<= IMG LEFT", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, "IMG RIGHT =>", (W - 250, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, f"peak {i+1}/{len(peaks)}   frame {real_frame}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "LEFT leg on which side?  1=img-left  2=img-right    ->/n next   s skip   q quit",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if H > MAX_DISPLAY_H:
            img = cv2.resize(img, (int(W * MAX_DISPLAY_H / H), MAX_DISPLAY_H))

        cv2.imshow("mark anchor - which SIDE is the LEFT leg?", img)
        key = cv2.waitKeyEx(0)
        low = key & 0xFF
        if key in NEXT_ARROWS or low in (ord("n"), ord("."), ord("d")):
            i = (i + 1) % len(peaks)                                          #show the next peak
        elif low == ord("1"):   #left leg is on the image-LEFT -> the smaller-x detection is the left leg
            result = (anchor, bool(a15[anchor, 0] < a16[anchor, 0])); break
        elif low == ord("2"):   #left leg is on the image-RIGHT -> the larger-x detection is the left leg
            result = (anchor, bool(a15[anchor, 0] > a16[anchor, 0])); break
        elif low == ord("s"):
            result = "skip"; break
        elif low in (ord("q"), 27):
            result = None; break

    cap.release()
    cv2.destroyAllWindows()
    return result


def main():
    fbf = json.load(open(FBF_PATH))
    strike = json.load(open(STRIKE_PATH))

    by_video = defaultdict(list)
    for r in fbf:
        by_video[r["video"]].append(r)

    corrected_fbf = []
    swaps_lookup = {}       # (base_video, frame) -> swapped bool, reused for the strikefoot pass
    anchors_record = {}     # video -> {anchor_frame, left_is_det0} : the record we MUST save

    for video, records in by_video.items():
        records.sort(key=lambda r: r["Frame"])
        frames = [r["Frame"] for r in records]
        kpts = np.array([r["Keypoints"] for r in records], dtype=float)   # (n,21,2), normalised
        a15, a16 = kpts[:, L_ANKLE, :], kpts[:, R_ANKLE, :]
        peaks = find_peak_anchors(a15, a16)
        key = base_name(video)

        vp = resolve_video(video)
        if vp is None:
            print(f"{video:24s} -> NO VIDEO FILE, left uncorrected")
            corrected_fbf.extend(records)
            for f in frames:
                swaps_lookup[(key, f)] = False
            continue

        ans = mark_orientation(vp, frames, peaks, a15, a16)
        if ans is None:
            print("quit - nothing written this run (no resume; re-run to start over)")
            return
        if ans == "skip":
            print(f"{video:24s} -> skipped, left uncorrected")
            corrected_fbf.extend(records)
            for f in frames:
                swaps_lookup[(key, f)] = False
            continue

        anchor, left_is_det0 = ans
        anchors_record[key] = {"anchor_frame": int(frames[anchor]), "left_is_det0": bool(left_is_det0)}

        swapped = continuity_from_anchor(a15, a16, anchor, left_is_det0)
        corrected = correct_sequence(kpts, swapped)
        for i, r in enumerate(records):
            corrected_fbf.append({"Frame": r["Frame"], "Keypoints": corrected[i].tolist(), "video": r["video"]})
            swaps_lookup[(key, r["Frame"])] = bool(swapped[i])
        print(f"{video:24s} -> anchor@frame {frames[anchor]} (peak of {len(peaks)}), "
              f"swapped {int(swapped.sum())}/{len(frames)}")

    #write outputs
    json.dump(corrected_fbf, open(OUT_FBF, "w"))
    json.dump(anchors_record, open(ANCHORS_PATH, "w"), indent=2)
    print(f"\nwrote {len(corrected_fbf)} corrected frame rows -> {OUT_FBF}")
    print(f"wrote anchor record ({len(anchors_record)} videos) -> {ANCHORS_PATH}")

    #apply the SAME swaps to the strikefoot (d-frame) keypoints
    corrected_strike = []
    n_swapped = 0
    for r in strike:
        kp = np.array(r["Keypoints"], dtype=float)
        if swaps_lookup.get((base_name(r["video"]), r["Frame"]), False):
            kp = kp[FLIP_IDX]
            n_swapped += 1
        corrected_strike.append({"Frame": r["Frame"], "Keypoints": kp.tolist(), "video": r["video"]})
    json.dump(corrected_strike, open(OUT_STRIKE, "w"))
    print(f"wrote {len(corrected_strike)} strikefoot rows ({n_swapped} swapped) -> {OUT_STRIKE}")


if __name__ == "__main__":
    main()
