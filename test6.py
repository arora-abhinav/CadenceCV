#New test: creating 2 state vectors that would determine both continuity and physical plausibility:
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


with open("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/normalised_keypoints.pkl", "rb") as file:
    data = pickle.load(file)

non_normalised_kpts = data["Non-Normalised Keypoints"]
valid_frames = data["Valid Frames"]

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
#Applying the same permutation to every frame
flipped_kpts = non_normalised_kpts[:, FLIP_IDX, :]

suspicious_position_frames = []
final_kpts = non_normalised_kpts.clone().detach()

#The state vectors derived from current YOLO keypoints
n_frames = len(non_normalised_kpts)
left_knee_position = non_normalised_kpts[:, 13, :]
right_knee_position = non_normalised_kpts[:, 14, : ]
left_ankle_position = non_normalised_kpts[:, 15, : ]
right_ankle_position = non_normalised_kpts[:, 16, : ]
left_x_ankle_velocity = np.gradient(left_ankle_position[:, 0])
right_x_ankle_velocity = np.gradient(right_ankle_position[: ,0])
left_x_knee_velocity = np.gradient(left_knee_position[: ,0])
right_x_knee_velocity = np.gradient(right_knee_position[:, 0])
left_y_knee_velocity  = np.gradient(left_knee_position[:, 1])
right_y_knee_velocity = np.gradient(right_knee_position[:, 1])
left_y_ankle_velocity  = np.gradient(left_ankle_position[:, 1])
right_y_ankle_velocity = np.gradient(right_ankle_position[:, 1])
position_state_vector = np.concatenate([left_knee_position, right_knee_position, left_ankle_position, right_ankle_position], axis = 1)
velocity_state_vector = np.stack(arrays = [left_x_knee_velocity, left_y_knee_velocity, right_x_knee_velocity, right_y_knee_velocity, left_x_ankle_velocity, left_y_ankle_velocity, right_x_ankle_velocity, right_y_ankle_velocity], axis=1)
state_vector = np.concatenate([position_state_vector, velocity_state_vector], axis=1)

print(position_state_vector.shape)
# The state vectors derived from flipped YOLO keypoints
f_left_knee_position   = flipped_kpts[:, 13, :]
f_right_knee_position  = flipped_kpts[:, 14, :]
f_left_ankle_position  = flipped_kpts[:, 15, :]
f_right_ankle_position = flipped_kpts[:, 16, :]
f_left_x_knee_velocity   = np.gradient(f_left_knee_position[:, 0])
f_right_x_knee_velocity  = np.gradient(f_right_knee_position[:, 0])
f_left_x_ankle_velocity  = np.gradient(f_left_ankle_position[:, 0])
f_right_x_ankle_velocity = np.gradient(f_right_ankle_position[:, 0])
f_left_y_knee_velocity   = np.gradient(f_left_knee_position[:, 1])
f_right_y_knee_velocity  = np.gradient(f_right_knee_position[:, 1])
f_left_y_ankle_velocity  = np.gradient(f_left_ankle_position[:, 1])
f_right_y_ankle_velocity = np.gradient(f_right_ankle_position[:, 1])

flipped_position_state_vector = np.concatenate([f_left_knee_position, f_right_knee_position, f_left_ankle_position, f_right_ankle_position],axis=1)
flipped_velocity_state_vector = np.stack([f_left_x_knee_velocity, f_left_y_knee_velocity, f_right_x_knee_velocity, f_right_y_knee_velocity, f_left_x_ankle_velocity, f_left_y_ankle_velocity, f_right_x_ankle_velocity, f_right_y_ankle_velocity], axis=1)
flipped_state_vector = np.concatenate([flipped_position_state_vector, flipped_velocity_state_vector], axis=1)

#Iterating from one (assuming the first frame is a simple ground truth)
#Initially setting the position state vector 

#Standardising the data
mean = state_vector.mean(axis=0)
std = state_vector.std(axis=0) + 1e-8
state_vector = (state_vector - mean) / std
flipped_state_vector = (flipped_state_vector - mean) / std

previous_state = state_vector[0]
#Keeping a track of differences
differences = []

for i in range(1, n_frames):
    current_YOLO_state = state_vector[i]
    flipped_YOLO_state = flipped_state_vector[i]

    dist_1_metrics = []
    dist_2_metrics = []
    #Computing distances:
    for j in range(len(previous_state)):
        dist_1_vector = previous_state[j] - current_YOLO_state[j]
        dist_1_metrics.append(dist_1_vector)
        dist_2_vector = previous_state[j] - flipped_YOLO_state[j]
        dist_2_metrics.append(dist_2_vector)

    final_dist_1 = np.linalg.norm(dist_1_metrics)
    final_dist_2 = np.linalg.norm(dist_2_metrics)

    differences.append(final_dist_1 - final_dist_2)
    #Assigning the new keypoints and the previous state
    if final_dist_1 <= final_dist_2:
        previous_state = current_YOLO_state
        final_kpts[i] = non_normalised_kpts[i]
    else:
        previous_state = flipped_YOLO_state
        final_kpts[i] = flipped_kpts[i]

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

incorrect_switches = visualise_corrected_keypoints("/Users/abhinavarora/Desktop/CadenceCV/Videos/Video18.mp4", valid_frames, final_kpts)

#Now plotting the keypoints

left_knee = final_kpts[:, 13, :]
right_knee = final_kpts[:, 14, :]
left_ankle = final_kpts[:, 15, :]
right_ankle = final_kpts[:, 16, :]

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Left Knee
axs[0, 0].plot(left_knee[:, 0], label="X")
axs[0, 0].plot(left_knee[:, 1], label="Y")
axs[0, 0].set_title("Left Knee")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Right Knee
axs[0, 1].plot(right_knee[:, 0], label="X")
axs[0, 1].plot(right_knee[:, 1], label="Y")
axs[0, 1].set_title("Right Knee")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Left Ankle
axs[1, 0].plot(left_ankle[:, 0], label="X")
axs[1, 0].plot(left_ankle[:, 1], label="Y")
axs[1, 0].set_title("Left Ankle")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Right Ankle
axs[1, 1].plot(right_ankle[:, 0], label="X")
axs[1, 1].plot(right_ankle[:, 1], label="Y")
axs[1, 1].set_title("Right Ankle")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.xlabel("Frame")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.plot(range(1, n_frames), differences, linewidth=1.5)
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Frame")
plt.ylabel("dist_keep - dist_swap")
plt.title("Difference Between Keep and Swap Continuity Costs")
plt.grid(True)

plt.tight_layout()
plt.show()

