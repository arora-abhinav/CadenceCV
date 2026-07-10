#This file will be used to write the inference metrics obtained from footnet_inference.py
from footnet_inference import frame_to_pred_dict, strikefoot_frames
import numpy as np

#Clauclating the average time in air per gait cycle and average time in ground contact
def flight_time_and_gct():
    #Grouping pairs of 1s and 0s from the frame_to_pred_dict to obtain a gait cycle
    frame_preds = []
    for frame in frame_to_pred_dict:
        frame_preds.append(frame_to_pred_dict[frame])
    
