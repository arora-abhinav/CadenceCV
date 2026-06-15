#Script for loading and extracting keypoints from a video
from ultralytics import YOLO
import cv2
import numpy as np

def extract_keypoints(video_dir):
    #Obtaining metadata for the video to calculate duration
    cap = cv2.VideoCapture(video_dir)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
    #stream=True now allows us to process the video frame by frame
    source = model(source='/Users/abhinavarora/Desktop/CadenceCV/Videos/vidssave.com How 5K running paces looks on a treadmill! 15 minutes 5K. 720P.mp4', show=False, conf=0.3, stream=True)
    #Result for each frame
    right_ankle_coords = []
    left_ankle_coords = []
    right_hip_coords = []
    left_hip_coords = []
    left_knee_coords = []
    right_knee_cords = []
    left_shoulder_coords = []
    right_shoulder_coords = []
    frames = []
    #Frame index
    i = 0
    for result in source:
        #Obtaining the first person's keypoints from the video
        if len(result.keypoints) == 0:
            continue
        #Extracting essential keypoints based off indices according to the YOLOv8 documentation
        kpts = result.keypoints.xy[0]
        left_ankle = kpts[15]
        right_ankle = kpts[16]
        left_hip = kpts[11]
        right_hip = kpts[12]
        left_knee = kpts[13]
        right_knee = kpts[14]
        right_shoulder = kpts[6]
        left_shoulder = kpts[5]
        right_ankle_coords.append(right_ankle)
        left_ankle_coords.append(left_ankle)
        right_hip_coords.append(right_hip)
        left_hip_coords.append(left_hip)
        left_knee_coords.append(left_knee)
        right_knee_cords.append(right_knee)
        left_shoulder_coords.append(left_shoulder)
        right_shoulder_coords.append(right_shoulder)

        i += 1
        frames.append(i)

    frames = np.array(frames)
    right_ankle_coords = np.array(right_ankle_coords)
    left_ankle_coords = np.array(left_ankle_coords)
    right_hip_coords = np.array(right_hip_coords)
    left_hip_coords = np.array(left_hip_coords)
    left_knee_coords = np.array(left_knee_coords)
    right_knee_cords = np.array(right_knee_cords)
    right_shoulder_coords = np.array(right_shoulder_coords)
    left_shoulder_coords = np.array(left_shoulder_coords)

    return {
    "right_ankle": right_ankle_coords,
    "left_ankle": left_ankle_coords,
    "right_hip": right_hip_coords,
    "left_hip": left_hip_coords,
    "left_knee": left_knee_coords,
    "right_knee": right_knee_cords,
    "right_shoulder": right_shoulder_coords,
    "left_shoulder": left_shoulder_coords,
    "Duration": duration,
    "Source": source,
    "Frame Count": int(frame_count)
    }