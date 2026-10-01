# Single-function left/right (side) correction for one video's keypoints.
# YOLO's left/right ankle labels flicker (wrong ~half the frames in a side view), which poisons BOTH the raw
# keypoint features AND the per-leg FootNet features fed to the LSTM/SVM. This labels the two legs for the
# WHOLE video at once (Viterbi) by continuity of the full leg chain, seeds the naming from a frame the USER
# marks, and FLIP_IDX-swaps the whole body on the frames where YOLO was flipped. correct_side() is the single
# entry point.
import numpy as np
import cv2
from scipy.signal import find_peaks

#full left<->right keypoint relabel (same list the model files use for mirroring)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
MAX_DISPLAY_H = 900
_NEXT_KEYS = {65363, 2555904, 63235}   #right-arrow keycodes across platforms; n/./d also work

#(left, right) index pairs tracked together: knee, ankle, and the 2 foot points. knees and ankles cross at
#DIFFERENT moments of the stride, so at any crossing some joints are still far apart and settle who is who.
#tracking the knee alone (the old way) had nothing to fall back on when the two knees overlapped
LEG_PAIRS = [(13, 14), (15, 16), (17, 18), (19, 20)]
#price for overriding YOLO's own label on a frame. YOLO is right most of the time, so a flip has to be worth
#more than this in continuity. 0 = pure continuity (relabels long stretches wrongly), very big = never correct.
#0.1 - 1 all worked on Video18 (labelled by hand), 0.25 sits in the middle of that
SWAP_COST = 0.25


def _viterbi_swaps(norm, swap_cost=SWAP_COST):
    #state s_t = 1 means YOLO's L/R is flipped at frame t. the cost of a frame is how far each leg's joints land
    #from a constant velocity prediction off the previous 2 frames, so the state is tracked as the PAIR
    #(s_{t-1}, s_t) -> 4 states. solved for the whole video at once, so one bad call at a crossing cant lock in a
    #wrong identity for the next 80 frames like the old frame by frame tracker did. returns the per-frame swap flag
    n = len(norm)
    left = np.stack([norm[:, l] for l, r in LEG_PAIRS], axis=1)     # (n, joints, 2)
    right = np.stack([norm[:, r] for l, r in LEG_PAIRS], axis=1)

    def legs(t, s):
        #(track A, track B) joints at frame t if the state is s
        return (right[t], left[t]) if s else (left[t], right[t])

    def miss(actual, predicted):
        return np.linalg.norm(actual - predicted, axis=1).sum()

    #cost[(s_prev, s_curr)] for frames 0,1 (no velocity yet, so plain distance)
    cost = {}
    for s0 in (0, 1):
        for s1 in (0, 1):
            A0, B0 = legs(0, s0)
            A1, B1 = legs(1, s1)
            cost[(s0, s1)] = miss(A1, A0) + miss(B1, B0) + swap_cost * (s0 + s1)

    backpointers = []
    for t in range(2, n):
        new_cost, back = {}, {}
        for s1 in (0, 1):
            for s2 in (0, 1):
                A1, B1 = legs(t - 1, s1)
                A2, B2 = legs(t, s2)
                best, best_s0 = np.inf, 0
                for s0 in (0, 1):
                    A0, B0 = legs(t - 2, s0)
                    c = cost[(s0, s1)] + miss(A2, 2*A1 - A0) + miss(B2, 2*B1 - B0)
                    if c < best:
                        best, best_s0 = c, s0
                new_cost[(s1, s2)] = best + swap_cost * s2
                back[(s1, s2)] = best_s0
        cost = new_cost
        backpointers.append(back)

    #walk back from the cheapest final pair
    s1, s2 = min(cost, key=cost.get)
    states = [s2, s1]
    for back in reversed(backpointers):
        s0 = back[(s1, s2)]
        states.append(s0)
        s1, s2 = s0, s1
    return np.array(states[::-1], dtype=bool)


def _find_anchors(a15, a16):
    #anchor candidates = local maxima of ankle x-separation (feet split wide, L/R unambiguous), frame order
    dist = np.abs(a15[:, 0] - a16[:, 0])
    peaks, _ = find_peaks(dist)
    return [int(p) for p in peaks] if len(peaks) else [int(np.argmax(dist[1:-1])) + 1]


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


def correct_side_auto(normalised_kpts):
    #Same viterbi correction but with NO anchor click, for building the training data across every labelled video
    #at once. The naming is whatever YOLO says on the MAJORITY of frames - the strike labels were made against YOLO's
    #naming, so keeping its majority keeps the labels lined up and only the flickered frames get flipped back.
    #Returns (corrected keypoints (n, 21, 2), per-frame swap flag)
    norm = np.asarray(normalised_kpts, dtype=float).reshape(-1, 21, 2)
    if len(norm) < 3:
        return norm.copy(), np.zeros(len(norm), dtype=bool)
    swapped = _viterbi_swaps(norm)
    if swapped.mean() > 0.5:
        swapped = ~swapped

    out = norm.copy()
    out[swapped] = out[swapped][:, FLIP_IDX, :]
    return out, swapped


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
    #need 3 frames for the constant velocity prediction, anything shorter just keeps YOLO's labels
    if len(norm) < 3:
        swapped = np.zeros(len(norm), dtype=bool)
    else:
        swapped = _viterbi_swaps(norm)
    #the viterbi only decides which frames are flipped RELATIVE to each other, the user's mark decides the naming.
    #at the anchor the corrected left leg is det0 when that frame isnt swapped (det1 when it is). if that doesnt
    #match what the user marked, the whole video is named the wrong way round -> flip every frame
    if left_is_det0 == swapped[anchor]:
        swapped = ~swapped

    out = norm.copy()
    out[swapped] = out[swapped][:, FLIP_IDX, :]                       #whole-body L<->R relabel on swapped frames
    return out