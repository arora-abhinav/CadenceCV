# Plots temporal per-leg features over the video, with your MANUAL strikefoot labels overlaid as dots.
#   1) SIGNED foot angle  : arctan2(dy,dx) of heel->toe, direction-canonicalised (~sinusoid per foot).
#   2) ACUTE foot angle   : |angle to horizontal| in [0,90] (direction-agnostic, loses sign).
#   3) KNEE FLEXION        : interior angle at the knee (hip-knee-ankle). ~180 = straight leg, smaller = bent.
#      it's a JOINT angle, so it's far less sensitive to the foot foreshortening that made 1) noisy.
# All are savgol-smoothed. A YOLO left/right SWITCH shows up as a curve jumping onto the OTHER leg's phase.
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

KPTS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl"
LABELS_PATH = "/Users/abhinavarora/Desktop/CadenceCV/video18_gait_labels.json"
#feet: 17=L toe 18=R toe 19=L heel 20=R heel   |   legs: 11/12 hips, 13/14 knees, 15/16 ankles
L_TOE, R_TOE, L_HEEL, R_HEEL = 17, 18, 19, 20
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
SAVGOL_WINDOW, SAVGOL_POLY = 7, 3


def detect_direction(K):
    sh = (K[:, 5, 0] + K[:, 6, 0]) / 2
    hp = (K[:, 11, 0] + K[:, 12, 0]) / 2
    return 1 if np.median(sh - hp) > 0 else -1


def signed_angle(vec, direction):
    return np.degrees(np.arctan2(vec[:, 1], direction * vec[:, 0]))


def acute_angle(vec):
    return np.degrees(np.arctan2(np.abs(vec[:, 1]), np.abs(vec[:, 0])))


def knee_flexion(hip, knee, ankle):
    #interior angle at the knee between the thigh (knee->hip) and shank (knee->ankle). arccos -> [0,180]:
    #~180 = leg straight (segments in line), smaller = knee more flexed.
    v1 = hip - knee
    v2 = ankle - knee
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def sm(x):
    return savgol_filter(x, SAVGOL_WINDOW, SAVGOL_POLY)


def main():
    r = pickle.load(open(KPTS_PATH, "rb"))
    vf = np.array(r["Valid Frames"])
    K = np.array(r["Normalised Keypoints"]).reshape(len(vf), 21, 2)
    direction = detect_direction(K)

    #1) signed foot angle, 2) acute foot angle
    lv, rv = K[:, L_TOE, :] - K[:, L_HEEL, :], K[:, R_TOE, :] - K[:, R_HEEL, :]
    l_sign, r_sign = sm(signed_angle(lv, direction)), sm(signed_angle(rv, direction))
    l_ac, r_ac = sm(acute_angle(lv)), sm(acute_angle(rv))
    #3) knee flexion
    l_knee = sm(knee_flexion(K[:, L_HIP, :], K[:, L_KNEE, :], K[:, L_ANKLE, :]))
    r_knee = sm(knee_flexion(K[:, R_HIP, :], K[:, R_KNEE, :], K[:, R_ANKLE, :]))

    #ground-truth strikefoot (d) frames per leg
    lab = json.load(open(LABELS_PATH))
    f2i = {int(f): i for i, f in enumerate(vf)}
    l_strk = [int(f) for f, v in lab["L"].items() if v == "d" and int(f) in f2i]
    r_strk = [int(f) for f, v in lab["R"].items() if v == "d" and int(f) in f2i]

    def overlay(ax, l_ang, r_ang):
        ax.scatter(l_strk, [l_ang[f2i[f]] for f in l_strk], color="tab:blue",
                   edgecolor="k", s=65, zorder=5, label="L strikefoot (truth)")
        ax.scatter(r_strk, [r_ang[f2i[f]] for f in r_strk], color="tab:orange",
                   edgecolor="k", s=65, marker="s", zorder=5, label="R strikefoot (truth)")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for ax, l, rr, name in [(ax1, l_sign, r_sign, "signed foot angle (deg)"),
                            (ax2, l_ac, r_ac, "acute foot angle (deg)"),
                            (ax3, l_knee, r_knee, "knee flexion (deg)")]:
        ax.plot(vf, l, color="tab:blue", lw=1.6, label="LEFT")
        ax.plot(vf, rr, color="tab:orange", lw=1.6, label="RIGHT")
        overlay(ax, l, rr)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", ncol=2)
    ax1.set_title(f"Per-leg temporal features, savgol-smoothed (dir={'right' if direction > 0 else 'left'})  "
                  f"- dots = ground-truth strikefoot")
    ax3.set_xlabel("video frame")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
