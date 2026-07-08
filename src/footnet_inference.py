from footnet_model import CustomDataLoader, LSTM_custom
import torch
from obtain_metrics import configure_data, compute_all_metrics
import pickle
from torch.utils.data import DataLoader

#Loading the model:
model = LSTM_custom(input_size=4, hidden_size=32, num_layers=1, num_classes=1)
model.load_state_dict(torch.load("/Users/abhinavarora/Desktop/CadenceCV/models/footnet_lstm_best.pth", weights_only=True))
model.eval()

video_dir = ''
#Computing metrics from the video:
frame_by_frame_data = compute_all_metrics('')
#Loading mean and std from pickle file

with open("/Users/abhinavarora/Desktop/CadenceCV/models/scaler_means_and_dev.pkl", "rb") as file:
    data = pickle.load(file)

means = data["Means"]
stds = data["Stds"]

#Running data through the CustomDataLoader
all_features, all_masks, all_frames = configure_data(frame_by_frame_data, means, stds)
dataset = CustomDataLoader(all_features, all_masks)

#Running data through torch's actual data loader:
video_data = DataLoader(dataset, batch_size=32, shuffle=True)

#Now running the inference loop.

all_predictions = []
with torch.no_grad():
    for (ind, data) in enumerate(video_data):
        #Compute the forward pass:
        features, masks = data
        logits = model(features)
        predictions = (torch.sigmoid(logits) > 0.35).long()

        #Only using the predictions where the mask is true:
        predictions = predictions[masks]
        all_predictions.append(predictions)