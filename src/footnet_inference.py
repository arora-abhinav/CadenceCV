from footnet_model import CustomDataLoader, LSTM_custom
import torch
from obtain_metrics import configure_data, compute_all_metrics
import pickle
from torch.utils.data import DataLoader
from scipy import stats
import cv2

#Loading the model:
model = LSTM_custom(input_size=4, hidden_size=32, num_layers=1, num_classes=1)
model.load_state_dict(torch.load("/Users/abhinavarora/Desktop/CadenceCV/models/footnet_lstm_best.pth", weights_only=True))
model.eval()

video_dir = '/Users/abhinavarora/Desktop/CadenceCV/Videos/Video17.mp4'
#Computing metrics from the video:
frame_by_frame_data = compute_all_metrics(video_dir)
#Loading mean and std from pickle file

with open("/Users/abhinavarora/Desktop/CadenceCV/models/scaler_means_and_dev.pkl", "rb") as file:
    data = pickle.load(file)

means = data["Means"]
stds = data["Stds"]

#Running data through the CustomDataLoader
all_features, all_masks, all_frames = configure_data(frame_by_frame_data, means, stds)
dataset = CustomDataLoader(all_features, all_masks)

batch_size = 32
#shuffle=False so batch i always maps to windows [i*batch_size, (i+1)*batch_size)
video_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#Inference loop — storing unmasked predictions (shape: batch_size x 40) so the
#vote loop can index per-window using i * batch_size
all_predictions = []
with torch.no_grad():
    for (ind, data) in enumerate(video_data):
        features, masks = data
        logits = model(features)
        predictions = (torch.sigmoid(logits) > 0.35).long()
        all_predictions.append(predictions)

#Sliding window majority vote where each frame collects predictions from every window that covered it
#The predictions given by the LSTM are batched, meaning that each batch is of shape (32, 40) and not 
#(32, 40, 1) due to output.squeeze done in footnet_model. This corresponds to 32 chunks of size 40 predictions (The resampling num)
#So, each batch must be iterated over where 32 chunks of size 40 are obtained.
#Instead of iterating over contiguous chunks, chunks in continguous batches are instead iterated over
frame_to_pred_dict = {}
for (i, batch_preds) in enumerate(all_predictions):
    #The first 40 frame window always will start at i * batch_size and the last 40 frame window will
    #be found at (i + 1) * batch_size - 1
    window_start = i * batch_size
    #Last batch may not have 32 chunks
    batch_size_actual = batch_preds.shape[0]

    #Now this is iterating over the actual 40-sized framed windows
    for j in range(batch_size_actual):
        window_idx = window_start + j
        #Apply mask to get only real (non-padded) frames and predictions for this window
        window_mask = all_masks[window_idx]
        #applying the specific mask to a specific window in this current batch
        window_frames = all_frames[window_idx][window_mask]
        window_preds = batch_preds[j][window_mask]

        #Finally constructing the frame_to_pred_dict : {frame: predictions over each window}
        for (ind, frame) in enumerate(window_frames):
            frame_int = int(frame.item())
            if frame_int not in frame_to_pred_dict:
                frame_to_pred_dict[frame_int] = []
            frame_to_pred_dict[frame_int].append(window_preds[ind].item())

#Taking a majiority vote via mode
for frame in frame_to_pred_dict:
    frame_to_pred_dict[frame] = stats.mode(frame_to_pred_dict[frame], keepdims=True).mode[0]

print(frame_to_pred_dict)

#Detecting where strikefoot is actually happening 
strikefoot_frames = []
prev_pred = 0
for f in sorted(frame_to_pred_dict.keys()):
    curr_pred = frame_to_pred_dict[f]
    if curr_pred == 1 and prev_pred == 0:
        strikefoot_frames.append(f)
    prev_pred = curr_pred

cap = cv2.VideoCapture(video_dir)
for f in strikefoot_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ret, img = cap.read()
    cv2.imshow("Strikefoot frame", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()