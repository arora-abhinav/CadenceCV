# ============================================================================
# WHAT THIS SCRIPT IS FOR
# ----------------------------------------------------------------------------
# YOLO's left/right ankle tags flicker (wrong ~half the frames in a side view).
# Continuity tracking fixes the FLICKER (keeps each physical foot on one track),
# but it can't know which track is anatomically LEFT - that's one bit per video,
# and if the seed frame is itself flipped the WHOLE video comes out flipped.
#
# So: seed from a frame we can trust. I pick the frame where the two ankles are
# MAXIMALLY separated in x (feet split fore/aft, no occlusion) via find_peaks,
# show it in OpenCV, and let the user mark which ankle is the left leg. From that
# anchor i run continuity FORWARD and BACKWARD, so the whole video inherits one
# correct, human-confirmed orientation.
#
# Tracking runs on the BBOX-NORMALISED coords (bbox-relative, comparable frame to
# frame); the OpenCV display uses the raw PIXEL coords so the dots land on the runner.
# ============================================================================
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

KPTS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl"
VIDEO_PATH = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4"
L_ANKLE, R_ANKLE = 15, 16
MAX_DISPLAY_H = 900   #Video18 is 1920 tall - shrink the on-screen copy to fit the monitor


def _assign(d0, d1, predA, predB):
    #the core pairing test, shared everywhere: hand the two detections (d0,d1) to tracks A,B whichever way
    #lands closer to where each track was PREDICTED to be. Returns (A_point, B_point, was_swapped).
    keep = np.linalg.norm(predA - d0) + np.linalg.norm(predB - d1)   #A<-d0, B<-d1
    swap = np.linalg.norm(predA - d1) + np.linalg.norm(predB - d0)   #A<-d1, B<-d0
    if swap < keep:
        return d1, d0, True
    return d0, d1, False


def sign_flips(diff):
    #count how many times (A_x - B_x) changes sign = how many times the two ankles cross in x.
    #real running crosses ~2x per stride; far above that = flickering. My before/after score.
    s = np.sign(diff); s[s == 0] = 1
    return int((s[:-1] != s[1:]).sum())


def find_anchor(a15, a16):
    #the anchor = a frame where the two ankles are far apart in x, so left vs right is unambiguous to mark.
    dist = np.abs(a15[:, 0] - a16[:, 0])          #fore-aft separation of the ankles, per frame
    peaks, _ = find_peaks(dist)                   #local maxima of separation
    #use the FIRST peak: it's early (so forward prop has room) and find_peaks never returns the first/last
    #sample, so it's guaranteed not to be the last two frames (which would starve the velocity seed).
    return int(peaks[0]) if len(peaks) else int(np.argmax(dist[1:-1])) + 1


def mark_anchor(frame_no, ankle0_px, ankle1_px):
    #show the anchor frame with the two ankle detections drawn, and ask which one is the LEFT leg.
    #returns True if detection-0 (YOLO's "15") is the left leg, False if detection-1 is.
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"couldnt read frame {frame_no}")

    #draw at full-res PIXEL coords, then resize the whole annotated image so scaling isnt needed on the dots
    for label, (px, py), color in [("1", ankle0_px, (255, 0, 0)), ("2", ankle1_px, (0, 140, 255))]:
        cv2.circle(img, (int(px), int(py)), 9, color, -1)
        cv2.putText(img, label, (int(px) + 12, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(img, "which dot is the LEFT leg?  press 1 or 2  (q=quit)", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h, w = img.shape[:2]
    if h > MAX_DISPLAY_H:
        img = cv2.resize(img, (int(w * MAX_DISPLAY_H / h), MAX_DISPLAY_H))

    while True:
        cv2.imshow("mark the anchor frame", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            left_is_det0 = True; break
        if key == ord("2"):
            left_is_det0 = False; break
        if key in (ord("q"), 27):
            cv2.destroyAllWindows(); raise SystemExit("aborted at marking")
    cv2.destroyAllWindows()
    return left_is_det0


def continuity_from_anchor(det0, det1, anchor, left_is_det0):
    #seed the tracks at the anchor with the user's marking (A = left leg), then propagate continuity BOTH ways.
    n = len(det0)
    A = np.zeros((n, 2)); B = np.zeros((n, 2)); swapped = np.zeros(n, bool)

    #seed: A is whichever detection the user said is the left leg
    if left_is_det0:
        A[anchor], B[anchor] = det0[anchor], det1[anchor]
    else:
        A[anchor], B[anchor] = det1[anchor], det0[anchor]
        swapped[anchor] = True   #YOLO's labelling was flipped at the anchor

    #FORWARD: anchor+1 .. end. Velocity guess once i have two prior assigned frames (t-2 >= anchor),
    #else the first step falls back to last-position (safe here: the anchor is a max-separation frame).
    for t in range(anchor + 1, n):
        if t - 2 >= anchor:
            predA, predB = 2 * A[t-1] - A[t-2], 2 * B[t-1] - B[t-2]
        else:
            predA, predB = A[t-1], B[t-1]
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)

    #BACKWARD: anchor-1 .. 0. Extrapolate from the two LATER (already-assigned) frames t+1, t+2.
    #Precaution: only use t+2/t+1 if those indices exist, else fall back to last-position.
    for t in range(anchor - 1, -1, -1):
        if t + 2 < n and t + 1 < n:
            predA, predB = 2 * A[t+1] - A[t+2], 2 * B[t+1] - B[t+2]
        else:
            predA, predB = A[t+1], B[t+1]
        A[t], B[t], swapped[t] = _assign(det0[t], det1[t], predA, predB)

    return A, B, swapped


def main():
    r = pickle.load(open(KPTS_PATH, "rb"))
    vf = np.array(r["Valid Frames"])                                   #real video frame numbers
    #NORMALISED coords (bbox-relative) drive the tracking - comparable frame to frame
    K = np.array(r["Normalised Keypoints"]).reshape(len(vf), 21, 2)
    a15, a16 = K[:, L_ANKLE, :], K[:, R_ANKLE, :]
    #RAW PIXEL coords are only for drawing the dots on the actual frame
    px = np.array(r["Non-Normalised Keypoints"])

    #1) pick the anchor (first max-separation peak) and 2) have the user orient it
    anchor = find_anchor(a15, a16)
    print(f"anchor frame (max ankle separation): index {anchor} -> video frame {int(vf[anchor])}")
    left_is_det0 = mark_anchor(vf[anchor], px[anchor, L_ANKLE], px[anchor, R_ANKLE])
    print("left leg =", "detection 15" if left_is_det0 else "detection 16", "at the anchor")

    #3) propagate continuity forward + backward from the anchor
    before = sign_flips(a15[:, 0] - a16[:, 0])
    A, B, swapped = continuity_from_anchor(a15, a16, anchor, left_is_det0)
    after = sign_flips(A[:, 0] - B[:, 0])
    print(f"x-sign-flips BEFORE (raw YOLO): {before}   AFTER (anchored continuity): {after}")
    print(f"frames un-swapped: {int(swapped.sum())}/{len(vf)} = {swapped.mean():.0%}")

    #--- plot before vs after; A is now the human-confirmed LEFT leg ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax0.plot(vf, a15[:, 0], color="tab:blue", lw=1, label="YOLO left-ankle x (15)")
    ax0.plot(vf, a16[:, 0], color="tab:orange", lw=1, label="YOLO right-ankle x (16)")
    ax0.axvline(vf[anchor], color="green", ls="--", lw=1, label="anchor")
    ax0.set_title(f"BEFORE - raw YOLO ({before} x-crossings, mostly fake swaps)")
    ax0.set_ylabel("ankle x"); ax0.legend(loc="upper right")

    ax1.plot(vf, A[:, 0], color="tab:blue", lw=1, label="track A = LEFT (confirmed)")
    ax1.plot(vf, B[:, 0], color="tab:orange", lw=1, label="track B = RIGHT")
    ax1.axvline(vf[anchor], color="green", ls="--", lw=1, label="anchor")
    ax1.set_title(f"AFTER - anchored continuity ({after} x-crossings)")
    ax1.set_ylabel("ankle x"); ax1.set_xlabel("video frame"); ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
