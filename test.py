#Continuity reassignment for the ankle tracks. YOLO's left/right label flickers at every crossing, so i
#IGNORE the label and just track the two ankle DETECTIONS: each frame, assign the two points to two
#persistent tracks (A/B) by whichever pairing keeps both trajectories smooth (min total jump vs a
#constant-velocity prediction). Genuine crossings survive; the spurious label-swaps get corrected.
import pickle
import numpy as np
import matplotlib.pyplot as plt

KPTS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl"
L_ANKLE, R_ANKLE = 15, 16


def sign_flips(diff):
    #how many times (trackA_x - trackB_x) changes sign = how many times the two ankles cross in x
    s = np.sign(diff); s[s == 0] = 1
    return int((s[:-1] != s[1:]).sum())


def continuity_tracks(det0, det1):
    #det0/det1: (n,2) the two ankle detections per frame (YOLO's 15 and 16, whose L/R label we DON'T trust).
    #Returns two tracks A,B that stay continuous, plus a flag per frame for where we had to un-swap YOLO.
    n = len(det0)
    A = np.zeros((n, 2)); B = np.zeros((n, 2)); swapped = np.zeros(n, bool)
    A[0], B[0] = det0[0], det1[0]
    for t in range(1, n):
        #constant-velocity guess of where each track should land this frame
        predA = 2 * A[t-1] - A[t-2] if t >= 2 else A[t-1]
        predB = 2 * B[t-1] - B[t-2] if t >= 2 else B[t-1]
        d0, d1 = det0[t], det1[t]
        keep = np.linalg.norm(predA - d0) + np.linalg.norm(predB - d1)   #A<-15, B<-16
        swap = np.linalg.norm(predA - d1) + np.linalg.norm(predB - d0)   #A<-16, B<-15
        if swap < keep:
            A[t], B[t] = d1, d0; swapped[t] = True
        else:
            A[t], B[t] = d0, d1
    return A, B, swapped


def main():
    r = pickle.load(open(KPTS_PATH, "rb"))
    vf = np.array(r["Valid Frames"])
    K = np.array(r["Normalised Keypoints"]).reshape(len(vf), 21, 2)
    a15, a16 = K[:, L_ANKLE, :], K[:, R_ANKLE, :]

    before = sign_flips(a15[:, 0] - a16[:, 0])
    A, B, swapped = continuity_tracks(a15, a16)
    after = sign_flips(A[:, 0] - B[:, 0])

    print(f"x-sign-flips BEFORE (YOLO raw labels): {before}")
    print(f"x-sign-flips AFTER  (continuity):      {after}")
    print(f"frames where continuity had to un-swap YOLO: {int(swapped.sum())} / {len(vf)} "
          f"= {swapped.mean():.0%}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax0.plot(vf, a15[:, 0], color="tab:blue", lw=1, label="YOLO left-ankle x (15)")
    ax0.plot(vf, a16[:, 0], color="tab:orange", lw=1, label="YOLO right-ankle x (16)")
    ax0.set_title(f"BEFORE - raw YOLO labels ({before} x-crossings, most are spurious swaps)")
    ax0.set_ylabel("ankle x"); ax0.legend(loc="upper right")

    ax1.plot(vf, A[:, 0], color="tab:blue", lw=1, label="track A x")
    ax1.plot(vf, B[:, 0], color="tab:orange", lw=1, label="track B x")
    #mark the frames we un-swapped so i can see where YOLO was wrong
    sw = np.flatnonzero(swapped)
    ax1.scatter(vf[sw], A[sw, 0], color="red", s=12, zorder=3, label="un-swapped here")
    ax1.set_title(f"AFTER - continuity tracks ({after} x-crossings ~= 2 per stride, spurious swaps removed)")
    ax1.set_ylabel("ankle x"); ax1.set_xlabel("video frame"); ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
