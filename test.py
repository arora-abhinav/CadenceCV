#Diagnostic for the left/right keypoint mess. Draws, on every frame, the EXACT keypoints the SVM sees -
#YOLO's pixel keypoints, normalised by the bounding box the way svm_inference does, then re-normalised
#(mapped straight back to pixel space) and drawn. That round-trip is lossless, so if they land wrong on the
#person the fault is upstream (keypoints/mapping), not obtain_which_leg's geometry. Colour = side
#(green=left, red=right) so a swap is obvious. Now also runs a STOCK yolo pose model side by side so i can
#tell whether the bad bounding box is my custom final_model.pt or something about the video/person itself.
import cv2
from ultralytics import YOLO

FINAL_MODEL = "/Users/abhinavarora/Desktop/CadenceCV/ml/weights/final_model.pt"
STOCK_MODEL = "yolo26x-pose.pt"   #stock COCO pose (17 kpts, no toe/heel) - ultralytics downloads it if missing
VIDEO_PATH = "/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4"

#keypoint layout. 11-16 (hips/knees/ankles) are shared by COCO and my custom model; 17-20 (toe/heel L/R)
#only exist on final_model.pt, so they simply wont appear for the stock 17-kpt model
NAMES = {
    11: "L hip",  12: "R hip",
    13: "L knee", 14: "R knee",
    15: "L ankle", 16: "R ankle",
    17: "L toe",  18: "R toe",
    19: "L heel", 20: "R heel",
}
LEFT_IDS = {5, 11, 13, 15, 17, 19}
RIGHT_IDS = {6, 12, 14, 16, 18, 20}


def side_colour(idx):
    #BGR. green for left-side joints, red for right-side, dim blue for the centre/face/arm ones i dont label
    if idx in LEFT_IDS:
        return (0, 255, 0)
    if idx in RIGHT_IDS:
        return (0, 0, 255)
    return (255, 128, 0)


def run_keypoint_check(model_path, video_path, window_title):
    model = YOLO(model_path)
    #imgsz=1280 to match how svm_inference runs it - keypoints/boxes shift if the inference size differs
    preds = model.predict(video_path, imgsz=1280, stream=True)

    for frame_idx, res in enumerate(preds):
        img = res.orig_img.copy()          #the actual frame this result came from
        H, W = img.shape[:2]

        if len(res) == 0:
            cv2.putText(img, f"frame {frame_idx}: NO DETECTION", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            #highest-confidence person, same as svm_inference picks
            kpts = res.keypoints.xy[0]                    #pixel xy, shape [n_kpts, 2]
            x1, y1, x2, y2 = res.boxes.xyxy[0].tolist()
            x_diff, y_diff = (x2 - x1), (y2 - y1)

            for idx, (px, py) in enumerate(kpts.tolist()):
                #normalise EXACTLY like svm_inference (bbox-relative), then re-normalise back to pixels.
                #lossless by construction, so what i draw IS what the model was fed
                xn = (px - x1) / x_diff if x_diff else 0.0
                yn = (py - y1) / y_diff if y_diff else 0.0
                xr = int(xn * x_diff + x1)
                yr = int(yn * y_diff + y1)

                colour = side_colour(idx)
                cv2.circle(img, (xr, yr), 4, colour, -1)
                if idx in NAMES:
                    cv2.circle(img, (xr, yr), 6, colour, 2)
                    cv2.putText(img, f"{idx} {NAMES[idx]}", (xr + 8, yr - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

            #the bounding box - bright + thick since this is the thing thats wrong. the normalisation is
            #RELATIVE to this box, so if the box doesnt cover the person the normalised coords are skewed
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            #quantify how much of the frame the box actually covers, so "doesnt encapsulate the person" is a number
            cov = f"box {100*x_diff/W:.0f}%W x {100*y_diff/H:.0f}%H of frame  |  {len(kpts)} kpts"
            cv2.putText(img, f"frame {frame_idx}  (green=left red=right)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, cov, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_title, img)
        #step frame by frame; q bails out of THIS model's run
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    #my custom model first (the suspect), then the stock model on the same video for comparison.
    #press q to finish one and move to the next
    run_keypoint_check(FINAL_MODEL, VIDEO_PATH, "final_model.pt  (custom, 21 kpts)")
    run_keypoint_check(STOCK_MODEL, VIDEO_PATH, "yolo26x-pose.pt  (stock, 17 kpts)")
