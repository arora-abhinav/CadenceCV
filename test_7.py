import numpy as np
import pickle
import cv2

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl", "rb") as f:
    data = pickle.load(f)

kpts = np.asarray(data["Non-Normalised Keypoints"])
valid_frames = data["Valid Frames"]

FLIP_IDX = [
    0,2,1,4,3,6,5,8,7,10,9,
    12,11,14,13,16,15,18,17,20,19
]

flipped_kpts = kpts[:, FLIP_IDX, :]

def extract_positions(pose_sequence):
    """
    Returns shape (n_frames, 12)

    [
        LKx,LKy,
        RKx,RKy,
        LAx,LAy,
        RAx,RAy,
        LEx,LEy,
        REx,REy
    ]
    """

    left_knee  = pose_sequence[:,13,:]
    right_knee = pose_sequence[:,14,:]

    left_ankle  = pose_sequence[:,15,:]
    right_ankle = pose_sequence[:,16,:]

    left_elbow  = pose_sequence[:,7,:]
    right_elbow = pose_sequence[:,8,:]

    return np.concatenate([
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        left_elbow,
        right_elbow
    ], axis=1)


pos = extract_positions(kpts)
flip_pos = extract_positions(flipped_kpts)

mean = pos.mean(axis=0)
std = pos.std(axis=0) + 1e-8

pos = (pos - mean) / std
flip_pos = (flip_pos - mean) / std

final_kpts = kpts.copy()

# Accepted state
previous_pos = pos[0]

previous_vel = np.zeros_like(previous_pos)

# weights
W_POSITION = 1.0
W_ACCELERATION = 0.5

flip_frames = []
differences = []

for i in range(1, len(pos)):

    candidates = [
        (pos[i],      kpts[i],         False),
        (flip_pos[i], flipped_kpts[i], True)
    ]

    best_cost = np.inf

    for candidate_pos, candidate_pose, flipped in candidates:

        # implied velocity
        velocity = candidate_pos - previous_pos

        # implied acceleration
        acceleration = velocity - previous_vel

        position_cost = np.linalg.norm(candidate_pos - previous_pos)

        acceleration_cost = np.linalg.norm(acceleration)

        total_cost = (
            W_POSITION * position_cost +
            W_ACCELERATION * acceleration_cost
        )

        if flipped:
            swap_cost = total_cost
        else:
            keep_cost = total_cost

        if total_cost < best_cost:
            best_cost = total_cost

            best_pos = candidate_pos
            best_pose = candidate_pose
            best_vel = velocity
            best_flip = flipped

    differences.append(keep_cost - swap_cost)

    previous_pos = best_pos
    previous_vel = best_vel

    final_kpts[i] = best_pose

    if best_flip:
        flip_frames.append(i)

print("Flip frames:", flip_frames)

#Visualising the frames:
def visualise_corrected_keypoints(video_path, vf, corrected_px):
    #step through EVERY frame showing the CORRECTED pixel keypoints, so i can eyeball whether the legs are
    #right after the flip. knees/ankles/feet get named labels + colours so left vs right is obvious.
    #space/n = next, b = back, q = quit. banner says whether THIS frame was flipped by the heuristic.
    incorrect_switches = []
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
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)        
            if idx in knee_names:                              
                cv2.circle(img, (x, y), 6, (0, 165, 255), -1)
                cv2.putText(img, knee_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 1)
            elif idx in ankle_names:                             #ankles yellow
                cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(img, ankle_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            elif idx in foot_names:                              #feet red
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, foot_names[idx], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        H, W = img.shape[:2]
        cv2.putText(img, str(differences[i]), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 130), 2)
        if H > 900:
            img = cv2.resize(img, (int(W * 900 / H), 900))
        cv2.imshow("corrected keypoints per frame", img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("n"), ord(" ")):
            i = min(i + 1, len(vf) - 1)
        elif key == ord("b"):
            i = max(i - 1, 0)
        elif key == ord('s'):
            incorrect_switches.append(vf[i])
        elif key  == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    return incorrect_switches

flip_mask = np.array(differences) > 0
final_kpts[1:][flip_mask] = final_kpts[1:][flip_mask][:, FLIP_IDX, :]

incorrect_switches = visualise_corrected_keypoints("/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4", valid_frames, final_kpts)