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
    n_detected = len(left_ankle_arr)
    frame_count = coords["Frame Count"]
    duration = coords["Duration"]

    # Ankle velocities
    left_ankle_x_vel = np.gradient(left_ankle_arr[:, 0])
    left_ankle_y_vel = np.gradient(left_ankle_arr[:, 1])
    right_ankle_x_vel = np.gradient(right_ankle_arr[:, 0])
    right_ankle_y_vel = np.gradient(right_ankle_arr[:, 1])

    # Shin vectors (ankle -> knee) and their x-velocity
    left_shin_vec = left_knee_arr  - left_ankle_arr
    right_shin_vec = right_knee_arr - right_ankle_arr
    left_shin_velocity = np.gradient(left_shin_vec[:, 0])
    right_shin_velocity = np.gradient(right_shin_vec[:, 0])

    def _tibial(shin_vec_arr):
        return np.pi / 2 - np.arctan2(shin_vec_arr[:, 1], shin_vec_arr[:, 0])

    left_tibial_angle = _tibial(left_shin_vec)
    right_tibial_angle = _tibial(right_shin_vec)

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
        },
    }

    results = []
  
    for side, d in legs.items():
        for frame in range(n_detected):
            ankle_arr = d['ankle_arr']
            knee_arr = d['knee_arr']
            hip_arr = d['hip_arr']
            shldr_arr = d['shldr_arr']

            v1 = hip_arr[frame] - knee_arr[frame]
            v2 = ankle_arr[frame] - knee_arr[frame]
            knee_flexion = float(
                np.rad2deg(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) % 360
            )

            offset = float(np.abs(ankle_arr[frame, 0] - hip_arr[frame, 0]))

            tv = shldr_arr[frame] - hip_arr[frame]
            trunk_angle = float(np.rad2deg(np.pi / 2 - np.arctan2(tv[1], tv[0])))

            pre2 = max(0, frame - 2)
            ankle_vel_pre = float(d['ay_vel'][pre2])

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
                'frame': frame,
                'side': side,
            })

    return results, frame_count, duration, right_hip_arr, left_hip_arr, right_ankle_arr, left_ankle_arr, right_knee_arr, left_knee_arr

#Mimics some functionality of the combined_labeller's configure_data. But, that script's data relied on distinguished gait cycles
#that were obtained via strikefoot data and then resampled to a size of 40 frames per gait cycle. Since that cannot happen anymore
#A sliding window of size 2 is being used. A window of size 40 with a step size of 2 across the frames and forms a usable input for
#for the LSTM.  
def configure_data(frame_by_frame_data: list[dict], scaler_means, scaler_stds):
    lstm_metrics = ["ankle_x_vel", "tibial_angle", "shin_velocity", "ankle_y_vel"]
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