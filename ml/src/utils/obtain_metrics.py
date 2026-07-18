import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.video_extracter import extract_keypoints
import torch
import pandas as pd 
from collections import deque
import cv2

def compute_all_metrics(video_dir: str):
    coords = extract_keypoints(video_dir)

    left_ankle_arr  = coords['left_ankle']
    right_ankle_arr = coords['right_ankle']
    left_hip_arr = coords['left_hip']
    right_hip_arr = coords['right_hip']
    left_knee_arr = coords['left_knee']
    right_knee_arr = coords['right_knee']
    left_shoulder_arr = coords['left_shoulder']
    right_shoulder_arr = coords['right_shoulder']
    frames = coords["Frames"]
    frame_count = coords["Frame Count"]
    duration = coords["Duration"]

    # These are two of the four (now 5) FootNet input features.
    left_ankle_x_vel = np.gradient(left_ankle_arr[:, 0])
    left_ankle_y_vel = np.gradient(left_ankle_arr[:, 1])
    right_ankle_x_vel = np.gradient(right_ankle_arr[:, 0])
    right_ankle_y_vel = np.gradient(right_ankle_arr[:, 1])

    # Shin vector: knee to ankle vector (shin bone)
    # This swing velocity is the third FootNet input feature.
    left_shin_vec = left_knee_arr  - left_ankle_arr
    right_shin_vec = right_knee_arr - right_ankle_arr
    left_shin_velocity = np.gradient(left_shin_vec[:, 0])
    right_shin_velocity = np.gradient(right_shin_vec[:, 0])

    def _tibial(shin_vec_arr):
        return np.pi / 2 - np.arctan2(shin_vec_arr[:, 1], shin_vec_arr[:, 0])

    left_tibial_angle = _tibial(left_shin_vec)
    right_tibial_angle = _tibial(right_shin_vec)

    # Auto-detect runner facing direction from hip X trajectory.
    # direction=+1 means running rightward (+X), -1 means leftward (-X).
# Runner facing direction: +1 = rightward, -1 = leftward
    mean_hip_x = (left_hip_arr[:, 0] + right_hip_arr[:, 0]) / 2
    direction = 1 if mean_hip_x[-1] > mean_hip_x[0] else -1

    # Signed inter-ankle distance — positive when THIS foot is ahead in direction of travel
    left_ankle_x_dist  = direction * (left_ankle_arr[:, 0] - right_ankle_arr[:, 0])
    right_ankle_x_dist = direction * (right_ankle_arr[:, 0] - left_ankle_arr[:, 0])


    legs = {
        'left': {
            'ax_vel': left_ankle_x_vel,
            'ay_vel': left_ankle_y_vel,
            'sh_vel': left_shin_velocity,
            'tib': left_tibial_angle,
            'ankle_arr': left_ankle_arr,
            'knee_arr': left_knee_arr,
            'hip_arr': left_hip_arr,
            'shldr_arr': left_shoulder_arr,
            'ankle_x_dist': left_ankle_x_dist
        },
        'right': {
            'ax_vel': right_ankle_x_vel,
            'ay_vel': right_ankle_y_vel,
            'sh_vel': right_shin_velocity,
            'tib': right_tibial_angle,
            'ankle_arr': right_ankle_arr,
            'knee_arr': right_knee_arr,
            'hip_arr': right_hip_arr,
            'shldr_arr': right_shoulder_arr,
            'ankle_x_dist': right_ankle_x_dist
        },
    }

    results = []
  
    for side, d in legs.items():
        for frame in frames:
            ankle_arr = d['ankle_arr']
            knee_arr = d['knee_arr']
            hip_arr = d['hip_arr']
            shldr_arr = d['shldr_arr']
            ankle_x_dist = d["ankle_x_dist"]

            # Knee flexion: v1 points from the knee toward the hip (along the thigh),
            # v2 points from the knee toward the ankle (along the shin). The difference
            # of their arctan2 angles gives the signed angle at the knee joint between
            # the two segments. % 360 maps it to [0, 360). A fully extended leg reads
            # ~180°; a bent knee reads less. At initial contact, overstriding produces
            # a stiffer, more extended knee (closer to 180°) because the leg is acting
            # as a rigid strut rather than a spring.
            v1 = hip_arr[frame] - knee_arr[frame]
            v2 = ankle_arr[frame] - knee_arr[frame]
            knee_flexion = float(
                np.rad2deg(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) % 360
            )

            # Ankle-hip horizontal offset: the absolute pixel distance between the ankle
            # and the hip along the x-axis. The hip is used as a COM proxy (greater
            # trochanter approximation). At initial contact this is the primary overstriding
            # metric from the literature — how far ahead of the COM the foot is landing.
            offset = float(np.abs(ankle_arr[frame, 0] - hip_arr[frame, 0]))

            # Trunk angle: tv is a vector pointing from the hip up toward the shoulder
            # along the torso. The same π/2 - arctan2 trick as _tibial re-references
            # the vector's angle from horizontal to vertical, giving forward trunk lean
            # in degrees. 0° = upright; positive = leaning forward. Forward lean affects
            # cadence, overstriding tendency, and running economy.
            tv = shldr_arr[frame] - hip_arr[frame]
            trunk_angle = float(np.rad2deg(np.pi / 2 - np.arctan2(tv[1], tv[0])))

            # Pre-strike ankle y-velocity: the vertical ankle speed 2 frames before
            # the current frame. Looking back 2 frames captures the descent phase just
            # before potential ground contact. In pixel coords positive = moving downward.
            # Heel strikers have a larger downward velocity here; forefoot strikers less.
            pre2 = max(0, frame - 2)
            ankle_vel_pre = float(d['ay_vel'][pre2])

            # Ankle approach angle: the direction of ankle travel over the 3 frames
            # immediately before this one, computed as arctan2 of the displacement vector.
            # Distinguishes a steep downward approach (heel strike, foot dropping from
            # above) from a more horizontal/rearward approach (forefoot, foot sweeping
            # back before contact). Clamped to frame 0 at the start of the video.
            p1 = max(0, frame - 1)
            p3 = max(0, frame - 3)
            dx = float(ankle_arr[p1, 0] - ankle_arr[p3, 0])
            dy = float(ankle_arr[p1, 1] - ankle_arr[p3, 1])
            approach_angle = float(np.arctan2(dy, dx))

            results.append({
                'ankle_x_vel': float(d['ax_vel'][frame]),
                'ankle_y_vel': float(d['ay_vel'][frame]),
                'shin_velocity': float(d['sh_vel'][frame]),
                'tibial_angle': float(d['tib'][frame]),
                'knee_flexion_angle': knee_flexion,
                'ankle_hip_horizontal_offset': offset,
                'trunk_angle': trunk_angle,
                'ankle_velocity_pre_strike': ankle_vel_pre,
                'ankle_approach_angle': approach_angle,
                'ankle_dist': float(ankle_x_dist[frame]),
                'frame': frame,
                'side': side,
            })

    return results, frame_count, duration, right_hip_arr, left_hip_arr, right_ankle_arr, left_ankle_arr, right_knee_arr, left_knee_arr

#Mimics some functionality of the combined_labeller's configure_data. But, that script's data relied on distinguished gait cycles
#that were obtained via strikefoot data and then resampled to a size of 40 frames per gait cycle. Since that cannot happen anymore
#A sliding window of size 2 is being used. A window of size 40 with a step size of 2 across the frames and forms a usable input for
#for the LSTM.  
def configure_data(frame_by_frame_data: list[dict], scaler_means, scaler_stds):
    lstm_metrics = ["ankle_x_vel", "tibial_angle", "shin_velocity", "ankle_y_vel", "ankle_dist"]
    resampling_num = 40
    stride = 5

    # Build video_metric_df from the dict directly
    video_metric_df = pd.DataFrame(frame_by_frame_data).dropna(axis=0, how='any').reset_index(drop=True)

    video_metric_df.loc[video_metric_df["side"] == "left", "side"] = "L"
    video_metric_df.loc[video_metric_df["side"] == "right", "side"] = "R"

    # Z-score normalisation using provided training stats (These were computed once in the previous configure_data 
    # function. Will be loaded from a pickle file)
    video_metric_df[lstm_metrics] = (
        video_metric_df[lstm_metrics] - scaler_means
    ) / scaler_stds

    all_features = []
    all_masks = []
    all_frames = []

    for side in ["L", "R"]:
        #Defining a subset based on leg and sorted frames
        subset = video_metric_df[
            video_metric_df["side"] == side
        ].sort_values("frame")[lstm_metrics].reset_index(drop=True)

        n_frames = len(subset)
        #This means there are no more frames to zero pad for this leg
        if n_frames < 1:
            continue

        start = 0
        while start < n_frames:
            end = start + resampling_num
            #Obtaining a chunk of data (size 40 max, will be resampled to size 40 via padding if not enough data)
            chunk = subset.iloc[start:end]
            chunk_len = len(chunk)
            mask_array = [True] * resampling_num
            #This array corresponds to each chunk's frames. Will be later used for a sliding window 
            #majiority vote
            chunk_frames = deque([i for i in range(start, start + chunk_len)])

            if chunk_len < resampling_num:
                pad_amount = resampling_num - chunk_len
                #Creating a padding amount (of zeros) based on remaining length to complete the chunk to give it a 
                #size 40
                padding = pd.DataFrame(
                    np.zeros((pad_amount, len(lstm_metrics))),
                    columns=lstm_metrics
                )

                for i in range(pad_amount):
                    mask_array[i] = False
                    #Sentinel -1 added to show that there is no dedicated frame for the zero padding part of the chunk
                    chunk_frames.appendleft(-1)
                    #Joining the padding and the chunk (padding first, then chunk) to ensure the 0s are initialy there
                    #conforming with the LSTM's data processing pipeline
                
                chunk = pd.concat([padding, chunk], ignore_index=True)

            all_features.append(chunk.to_numpy())
            all_masks.append(mask_array)
            all_frames.append(chunk_frames)
            #Going over the next 40 frames, skipping 2 beginning frames.
            start += stride

    all_features = torch.from_numpy(np.array(all_features)).float()
    all_masks = torch.from_numpy(np.array(all_masks))
    all_frames = torch.tensor(all_frames)

    return all_features, all_masks, all_frames