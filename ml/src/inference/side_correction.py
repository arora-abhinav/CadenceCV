# Single-function left/right (side) correction for one video's keypoints.
# YOLO's left/right ankle labels flicker (wrong ~half the frames in a side view), which poisons BOTH the raw
# keypoint features AND the per-leg FootNet features fed to the SVM. This tracks the two ankle DETECTIONS by
# continuity (ignoring YOLO's label), seeds the naming from a frame the USER marks, and FLIP_IDX-swaps the
# whole body on the frames where YOLO was flipped. correct_side() is the single entry point.
import numpy as np
import cv2
from scipy.signal import find_peaks

#full left<->right keypoint relabel (same list the model files use for mirroring)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
MAX_DISPLAY_H = 900
_NEXT_KEYS = {65363, 2555904, 63235}   #right-arrow keycodes across platforms; n/./d also work


def _assign(d0, d1, predA, predB):
    #hand the two detections to tracks A,B whichever pairing lands closer to prediction -> (A,B,swapped)
    keep = np.linalg.norm(predA - d0) + np.linalg.norm(predB - d1)
    swap = np.linalg.norm(predA - d1) + np.linalg.norm(predB - d0)
    if swap < keep:
        return d1, d0, True
    return d0, d1, False


def _find_anchors(a15, a16):
    #anchor candidates = local maxima of ankle x-separation (feet split wide, L/R unambiguous), frame order
    dist = np.abs(a15[:, 0] - a16[:, 0])
    peaks, _ = find_peaks(dist)
    return [int(p) for p in peaks] if len(peaks) else [int(np.argmax(dist[1:-1])) + 1]


def _continuity(det0, det1, anchor, left_is_det0):
    #seed A=left at the anchor, propagate continuity forward + backward, return the per-frame swap flag
    n = len(det0)
    A = np.zeros((n, 2)); B = np.zeros((n, 2)); swapped = np.zeros(n, bool)
    if left_is_det0:
        A[anchor], B[anchor] = det0[anchor], det1[anchor]
    else:
        A[anchor], B[anchor] = det1[anchor], det0[anchor]; swapped[anchor] = True
    for t in range(anchor + 1, n):
        predA, predB = (2*A[t-1]-A[t-2], 2*B[t-1]-B[t-2]) if t - 2 >= anchor else (A[t-1], B[t-1])
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)
    for t in range(anchor - 1, -1, -1):
        predA, predB = (2*A[t+1]-A[t+2], 2*B[t+1]-B[t+2]) if t + 2 < n else (A[t+1], B[t+1])
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)
    return swapped


def _mark_anchor(video_directory, peaks, px_kpts):
    #cv2 loads the video straight from video_directory. show the anchor frame with the two ankle PIXEL dots
    #(un-normalised via the bbox), cycle with ->/n, press 1/2 for which ankle is the LEFT leg. seeks by index
    #(the pickle's valid frames are 0-indexed & contiguous, so index == frame). returns (anchor, left_is_det0)|None
    cap = cv2.VideoCapture(video_directory)
    i = 0; result = None
    while True:
        anchor = peaks[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
        ret, img = cap.read()
        if not ret:
            i = (i + 1) % len(peaks); continue
        for lbl, idx, col in [("1", 15, (255, 0, 0)), ("2", 16, (0, 140, 255))]:
            x, y = int(px_kpts[anchor, idx, 0]), int(px_kpts[anchor, idx, 1])
            cv2.circle(img, (x, y), 9, col, -1)
            cv2.putText(img, lbl, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        cv2.putText(img, f"peak {i+1}/{len(peaks)}  frame {anchor}   1/2 = LEFT leg   ->/n next   q quit",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        h, w = img.shape[:2]
        if h > MAX_DISPLAY_H:
            img = cv2.resize(img, (int(w * MAX_DISPLAY_H / h), MAX_DISPLAY_H))
        cv2.imshow("mark anchor - which ankle is the LEFT leg?", img)
        key = cv2.waitKeyEx(0); low = key & 0xFF
        if key in _NEXT_KEYS or low in (ord("n"), ord("."), ord("d")):
            i = (i + 1) % len(peaks)
        elif low == ord("1"): result = (anchor, True); break
        elif low == ord("2"): result = (anchor, False); break
        elif low in (ord("q"), 27): result = None; break
    cap.release()
    cv2.destroyAllWindows()
    return result


def correct_side(video_directory, bbox_coords, normalised_kpts):
    #video_directory : path to the video file, loaded straight into cv2 for the marking display.
    #bbox_coords     : (n, 4) per-frame [x1,y1,x2,y2], to un-normalise the keypoints to pixels for the display.
    #normalised_kpts : (n, 21, 2) bbox-normalised keypoints - tracked, corrected, and returned.
    #Returns the CORRECTED normalised keypoints, shape (n, 21, 2).
    norm = np.asarray(normalised_kpts, dtype=float).reshape(-1, 21, 2)
    bbox = np.asarray(bbox_coords, dtype=float)                       # (n, 4)
    #un-normalise to pixels for the display: px = xy1 + norm * (xy2 - xy1)
    xy1 = bbox[:, None, :2]
    wh = bbox[:, None, 2:] - bbox[:, None, :2]
    px = xy1 + norm * wh                                              # (n, 21, 2) pixel keypoints

    #Seeing left and right knee instead of left and right ankle since knee seems to be the most reliable metric there is 
    a15, a16 = norm[:, 13, :], norm[:, 14, :]
    peaks = _find_anchors(a15, a16)
    marked = _mark_anchor(video_directory, peaks, px)
    if marked is None:
        return norm                                                  #aborted -> return uncorrected
    anchor, left_is_det0 = marked
    swapped = _continuity(a15, a16, anchor, left_is_det0)

    out = norm.copy()
    out[swapped] = out[swapped][:, FLIP_IDX, :]                       #whole-body L<->R relabel on swapped frames
    return out