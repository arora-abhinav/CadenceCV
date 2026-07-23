#This file will be used to write the inference metrics obtained from svm_inference.py. 
import numpy as np
import scipy.signal
import cv2
import torch

#Clauclating the average time in air per gait cycle and average time in ground contact
#kpts is NOT normalised
def calculate_metrics(strikefoot_frames:list, toe_off_frames:list, duration, detected_frames, cap: cv2.VideoCapture, kpts:torch.Tensor):
    flight_times = []
    gct_times = []

    right_hip_arr = []
    left_hip_arr = []
    for k in kpts.tolist():
        right_hip_arr.append(k[12])
        left_hip_arr.append(k[11])

    strikefoot_frames_tagged = [(x, "D") for x in strikefoot_frames]
    toe_off_frames_tagged = [(x,"U") for x in toe_off_frames]

    total_frames = strikefoot_frames + toe_off_frames
    total_frames.sort()

    #Due to YOLO missing out on some frames, the actual frame count is required so that the ratio of 
    #detected:actual frame count can be taken
    actual_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    detected_frame_count = len(detected_frames)
    for takeoff in toe_off_frames:
        #Find the next landing after this takeoff. Finding all landings initially to check if there even is a landing 
        #after the take off frames.
        next_landings = strikefoot_frames[strikefoot_frames > takeoff]
        if len(next_landings) > 0:
            next_landing = next_landings[0]
            flight_times.append(next_landing - takeoff)
    
    for landing in strikefoot_frames:
        next_takeoffs = toe_off_frames[toe_off_frames > landing]
        if len(next_takeoffs) > 0:
            next_takeoff = next_takeoffs[0]  
            gct_times.append(next_takeoff - landing)

    #Divide by detected frame count to obtain metrics in seconds
    average_flight_time = (np.mean(flight_times) if flight_times else 0) / detected_frame_count
    average_gct = (np.mean(gct_times) if gct_times else 0) / detected_frame_count

    #Cadence is the steps per second
    cadence = len(strikefoot_frames) / duration * (detected_frame_count/int(actual_frame_count))

    #Vertical oscillation: 
    # 1) Calculate the hip's mean Y coordinate (mean of right and left hip Y) over all frames
    # 2) Apply the Savitz-Golay smoothing filter (finetune the window size)
    # 3) Separate out the smoothed array based on each gait cycle: (continuous pairs of 0s and 1s)
    # 4) Obtain the vertical difference between each peak and trough (since pixel coordinates go downward)

    # Index 1 is the Y coordinate (vertical axis in pixel space)
    n = min(len(right_hip_arr), len(left_hip_arr))
    mean_hip_y_coords = []
    for i in range(n):
        mean_hip_y_coords.append(np.mean([right_hip_arr[i][1], left_hip_arr[i][1]]))
    
    #The window length is the number of coefficients for smoothing
    #The poly order is the degree of the filtering polynomial
    #Finetune these accordingly
    mean_hip_y_coords = scipy.signal.savgol_filter(mean_hip_y_coords, window_length=4, polyorder=3)

    #Extracting gait cycles: Pictorial explanation in the next cell:
    changes = np.where(np.diff(frame_preds) == -1)[0] + 1
    #Pairs of -1 is one gait cycle
    cycles = []
    cycle = []
    for i in changes:
        cycle.append(i)
        if len(cycle) == 2:
            cycles.append(cycle.copy())
            cycle = []
    
    oscillations = []
    for c in cycles:
        start = c[0]
        end = c[1]
        peak = max(mean_hip_y_coords[start:end])
        trough = min(mean_hip_y_coords[start:end])
        oscillations.append(peak - trough)
    
    avg_oscillation = np.mean(oscillations) if oscillations else 0
    
    #Calculating stride length: