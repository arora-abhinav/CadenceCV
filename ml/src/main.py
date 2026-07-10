import sys, os
# Add ml/src/ to path so package imports (utils., inference., models.) all resolve
sys.path.insert(0, os.path.dirname(__file__))

import pickle
from utils.obtain_metrics import compute_all_metrics
from inference.footnet_inference import run_inference

# Change video_dir and cache_path here when switching to a different video.
# Delete the cache file whenever you change video_dir. The cache is video-specific.
video_dir = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4"
cache_path = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/keypoints_cache.pkl"

if __name__ == "__main__":
    # Step 1: Run YOLO pose estimation once and cache the result.
    # On subsequent runs the cache is loaded directly
    # This means changes to run_inference or downstream metrics only cost LSTM time, not YOLO time.
    if not os.path.exists(cache_path):
        print("No cache found — running YOLO inference (this may take a minute)...")
        frame_by_frame_data, frame_count, duration, right_hip_arr, left_hip_arr, right_ankle_arr, left_ankle_arr, right_knee_arr, left_knee_arr = compute_all_metrics(video_dir)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "frame_by_frame_data": frame_by_frame_data,
                "frame_count": frame_count,
                "duration": duration,
                "right_hip_arr": right_hip_arr,
                "left_hip_arr": left_hip_arr,
                "right_ankle_arr": right_ankle_arr,
                "left_ankle_arr": left_ankle_arr,
                "right_knee_arr": right_knee_arr,
                "left_knee_arr": left_knee_arr,
            }, f)
    else:
        # Loading from cache
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        frame_by_frame_data = cache["frame_by_frame_data"]
        frame_count = cache["frame_count"]
        duration = cache["duration"]
        right_hip_arr = cache["right_hip_arr"]
        left_hip_arr = cache["left_hip_arr"]
        right_ankle_arr = cache["right_ankle_arr"]
        left_ankle_arr = cache["left_ankle_arr"]
        right_knee_arr = cache["right_knee_arr"]
        left_knee_arr = cache["left_knee_arr"]

    # Step 2: Run FootNet LSTM inference on the pre-computed per-frame metrics.
    # YOLO does NOT run inside run_inference — it only consumes frame_by_frame_data.
    strikefoot_frames, frame_to_pred_dict = run_inference(frame_by_frame_data)
    print(f"Detected {len(strikefoot_frames)} strikefoot frames: {strikefoot_frames}")

    # Step 3: Compute biomechanical metrics from the contact signal.
    # inference_metrics.py will be built here — GCT, cadence, VO, contact angles, etc.
    # metrics = compute_metrics(frame_to_pred_dict, strikefoot_frames, duration, frame_count,
    # right_hip_arr, left_hip_arr)
