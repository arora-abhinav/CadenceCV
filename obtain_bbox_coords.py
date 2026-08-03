#Records the per-frame bounding box for every video, then zips it into the frame-by-frame keypoint data.
#Why: the stored keypoints are BBOX-normalised, so without the box i cant un-normalise them back to pixels
#(needed to draw the ankles on the actual frame for the side-correction marking). This is a FULL YOLO pass
#with final_model.pt - meant for the RunPod GPU, not local. scp the Videos/ over first, adjust the paths.
import json
import os

from ultralytics import YOLO

#--- adjust these for the RunPod box after you scp the files over ---
FBF_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/frame_by_frame_keypoint_data.json"
VIDEOS_DIR = "/Users/abhinavarora/Desktop/CadenceCV/Videos"
MODEL_PATH = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/final_model.pt"
OUT_BBOX = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/bbox_coords.json"
OUT_MERGED = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/frame_by_frame_keypoint_data_with_bbox.json"

EXTENSIONS = [".mp4", ".MOV"]
IMGSZ = 1280   #match the rest of the pipeline. the bbox comes back in ORIGINAL frame pixels regardless of this


def base_name(v):
    return v.split(".")[0]


def main():
    data = json.load(open(FBF_PATH))

    #1) collect the unique videos out of the frame rows
    videos = set()
    for r in data:
        videos.add(r["video"])

    model = YOLO(MODEL_PATH)
    #video -> {"bboxes": [ {frame_idx: [x1,y1,x2,y2]}, ... ]}
    bbox_data = {}

    for video in videos:
        try:
            #2) find the actual file by trying each extension (the check + the inference both live in the try)
            path = None
            for ext in EXTENSIONS:
                cand = os.path.join(VIDEOS_DIR, base_name(video) + ext)
                if os.path.exists(cand):
                    path = cand
                    break
            if path is None:
                raise FileNotFoundError(f"no video file for {video}")

            #3) full inference; grab the top box's xyxy each frame (skip frames with no detection, same as
            #the keypoint extraction, so the frame indices line up)
            preds = model.predict(path, imgsz=IMGSZ, stream=True)
            frame_boxes = []
            for idx, res in enumerate(preds):
                if len(res) == 0:
                    continue
                xyxy = res.boxes.xyxy[0].tolist()   #.tolist() so its json-serialisable
                #the original frame_by_frame data is 1-INDEXED (cv2/enumerate is 0-based, but the extraction
                #stored idx+1), so i add 1 here too - otherwise the zip below is off by one and nothing matches
                frame_boxes.append({idx + 1: xyxy})   #key = 1-based frame, value = [x1,y1,x2,y2]
            bbox_data[video] = {"bboxes": frame_boxes}
            print(f"{video:24s} -> {len(frame_boxes)} bboxes")
        except Exception as e:
            print(f"{video:24s} -> SKIP ({e})")

    json.dump(bbox_data, open(OUT_BBOX, "w"))
    print(f"\nwrote raw bbox dict -> {OUT_BBOX}")

    #4) zip the bboxes into the frame rows, matched on (video, frame). flatten the per-video arrays into a
    #(video, frame) -> xyxy lookup first, then tag each frame row with its box.
    lookup = {}
    for video, d in bbox_data.items():
        for fd in d["bboxes"]:
            for f, xyxy in fd.items():
                lookup[(video, int(f))] = xyxy

    matched = 0
    for r in data:
        xyxy = lookup.get((r["video"], r["Frame"]))
        if xyxy is not None:
            r["bbox"] = xyxy
            matched += 1

    json.dump(data, open(OUT_MERGED, "w"))
    print(f"merged bbox into {matched}/{len(data)} frame rows -> {OUT_MERGED}")


if __name__ == "__main__":
    main()
