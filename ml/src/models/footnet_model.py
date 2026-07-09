import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from training.data_loader import training_df, testing_df
from scipy.stats import zscore
from collections import deque
import numpy as np
import json
import pandas as pd
import torch
import pickle

def configure_data(strike_df, frame_by_frame_path, fit_scaler=True, scaler_means=None, scaler_stds=None):
    """
    strike_df: dataframe of labeled strikes (your training_df or test equivalent)
    frame_by_frame_path: path to frame_by_frame_data.json
    fit_scaler: True for training data, False for test data
    scaler_means, scaler_stds: pass in training stats when fit_scaler=False
    """
    lstm_metrics = ["ankle_x_vel", "tibial_angle", "shin_velocity", "ankle_y_vel"]
    footnet_metrics = lstm_metrics + ["u_frame", "video", "frame", "side"]

    #Dropping the rows where the u_frame doesn't exist (That is the only thing that doesn't exist)
    strike_df = strike_df.dropna(axis=0, how='any').copy()

    #First step: Separate the strike_df by video into different dfs
    dfs_to_merge = []
    for v in strike_df["video"].unique():
        df = strike_df[strike_df["video"] == v][footnet_metrics].copy()
        #Now, sorting items by leg, then frame. This is important to label contact and non-contact strikes effectively
        df.sort_values(by=["side", "frame"], ascending=True, inplace=True)
        dfs_to_merge.append(df)

    footnet_training_df = pd.concat(dfs_to_merge, axis=0).reset_index(drop=True)

    #Now, obtaining the data from frame_by_frame_data.json
    with open(frame_by_frame_path, "r") as file:
        data = json.load(file)

    video_metric_df = pd.concat(
        [pd.DataFrame(data[key]) for key in data], axis=0
    ).dropna(axis=0, how='any').copy()

    video_metric_df.loc[video_metric_df["side"] == "left", "side"] = "L"
    video_metric_df.loc[video_metric_df["side"] == "right", "side"] = "R"

    #Applying z_score normalisation to specific columns in video_metric_df. In accordance with FootNet's research
    #fit_scaler=True for training data, False for test data (use training stats)
    if fit_scaler:
        scaler_means = video_metric_df[lstm_metrics].mean()
        scaler_stds = video_metric_df[lstm_metrics].std()

    video_metric_df[lstm_metrics] = (
        video_metric_df[lstm_metrics] - scaler_means
    ) / scaler_stds

    #Now, adding a column to video_metric_df to indicate where there is contact and no contact between the ground
    #Non-contact points: each row in footnet_training_df's u and d frames (between those frames)
    #Contact points: done via obtaining the previous same-side row's d frame and the current row's u frame
    #Initialsing label values to None -> Indicates there isn't enough info to fill in contact or no contact
    video_metric_df["label"] = None
    gait_cycle_boundaries = []

    for i, (index, row) in enumerate(footnet_training_df.iterrows()):
        # Non-contact points
        video_metric_df.loc[
            (video_metric_df["video"] == row["video"]) &
            (video_metric_df["frame"] >= row["u_frame"]) &
            (video_metric_df["frame"] < row["frame"]) &
            (video_metric_df["side"] == row["side"]),
            "label"
        ] = 0

        # Contact points — find previous strike of same video + same side
        same_side_prev = footnet_training_df[
            (footnet_training_df["video"] == row["video"]) &
            (footnet_training_df["side"] == row["side"]) &
            (footnet_training_df["frame"] < row["frame"])
        ]

        if not same_side_prev.empty:
            prev_row = same_side_prev.iloc[-1]
            video_metric_df.loc[
                (video_metric_df["video"] == row["video"]) &
                (video_metric_df["frame"] >= prev_row["frame"]) &
                (video_metric_df["frame"] < row["u_frame"]) &
                (video_metric_df["side"] == row["side"]),
                "label"
            ] = 1

            # Capture the full cycle boundary directly here
            gait_cycle_boundaries.append({
                "video": row["video"],
                "side": row["side"],
                "start_frame": prev_row["frame"],
                "end_frame": row["frame"] - 1
            })

    # Remove cycles where end_frame < start_frame (invalid)
    valid_cycles = [c for c in gait_cycle_boundaries if c['end_frame'] >= c['start_frame']]

    Q1 = 19
    Q3 = 27
    IQR = Q3 - Q1  # 8
    upper_bound = Q3 + 1.5 * IQR  # 27 + 12 = 39
    lower_bound = max(1, Q1 - 1.5 * IQR)  # at least 1 frame

    filtered_cycles = [c for c in valid_cycles if lower_bound <= (c['end_frame'] - c['start_frame']) <= upper_bound]

    #Now, obtaining the rows for each cycle:
    video_metric_df = video_metric_df.reset_index(drop=True)
    actual_cycles = []

    for cycle in filtered_cycles:
        df = video_metric_df[
            (video_metric_df["video"] == cycle["video"]) &
            (video_metric_df["side"] == cycle["side"]) &
            (video_metric_df["frame"] >= cycle["start_frame"]) &
            (video_metric_df["frame"] <= cycle["end_frame"])
        ].copy()
        #Sorting to ensure frames are in chronological order
        df.sort_values(by=["frame"], axis=0, inplace=True)
        labels = deque(df["label"].tolist())
        actual_cycles.append((df[lstm_metrics], labels))

    #The resampling num, which is the number of timesteps the strikefoot detection LSTM will have
    resampling_num = 40

    #iterate again to resample to size resampling_num (in this case 40)
    #Mask array required to tell which ones are 0 padded: for both the labels and the gait_cycle
    all_features = []
    all_masks = []
    all_labels = []

    for gait_cycle, labels in actual_cycles:
        gait_cycle = gait_cycle.copy()
        gait_cycle_len = len(gait_cycle)
        mask_array = [True] * resampling_num
        pad_amount = resampling_num - gait_cycle_len

        for i in range(pad_amount):
            #Zeros for each feature
            mask_array[i] = False
            gait_cycle.loc[-1] = [0, 0, 0, 0]
            labels.appendleft(-1)
            gait_cycle.index += 1
            gait_cycle.sort_index(inplace=True)
            gait_cycle.reset_index(drop=True, inplace=True)

        all_features.append(gait_cycle.to_numpy())
        all_labels.append(list(labels))
        all_masks.append(mask_array)

    #Finally converting to a tensor:
    all_features = torch.from_numpy(np.array(all_features)).float()
    all_labels = torch.from_numpy(np.array(all_labels))
    all_masks = torch.from_numpy(np.array(all_masks))

    return all_features, all_labels, all_masks, scaler_means, scaler_stds

# Training
train_features, train_labels, train_masks, means, stds = configure_data(
    training_df, "/Users/abhinavarora/Desktop/CadenceCV/ml/data/frame_by_frame_data.json", fit_scaler=True
)

# Test — pass back the training stats
test_features, test_labels, test_masks, _, _ = configure_data(
    testing_df, "/Users/abhinavarora/Desktop/CadenceCV/ml/data/frame_by_frame_data.json", fit_scaler=False,
    scaler_means=means, scaler_stds=stds
)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/weights/scaler_means_and_dev.pkl", "wb") as file:
    pickle.dump({"Means": means, "Stds": stds}, file)


#labels are required to be None since at time of inference, we have no labels obviously. But labels are still
#required for training the model and for testing it as below to check for accuracy
class CustomDataLoader(Dataset):
    def __init__(self, features, masks, labels = None):
        super(CustomDataLoader, self).__init__()
        #Loading the data from the tensor
        self.features:torch.Tensor = features
        self.labels:torch.Tensor = labels
        self.masks:torch.Tensor = masks
    def __getitem__(self, index):
        if self.labels != None:
            return self.features[index].float(), self.labels[index].float(), self.masks[index].bool()
        else:
            return self.features[index].float(), self.masks[index].bool()

    def __len__(self):
        return self.features.shape[0]

dataset = CustomDataLoader(train_features, train_labels, train_masks)
test_dataset = CustomDataLoader(test_features, test_labels, test_masks)
first_features, first_labels, first_masks = dataset[0]

#batch_side loads in 4 batches at a time, shuffle allows to randomly shuffle the batches
train_data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

#Finally constructing the LSTM to detect strikefoot 
#Input would be a tensor with shape: 
#(batch size (an X number of gait cycles will be fed at a time), A sequence length (the no. of hidden states), Input size = number of features there exist)
#Hidden size is the length of each hidden state vector

class LSTM_custom(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_custom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        #Input will have the following shape (32, 40, 4) for 32 gait cycles, 40 timesteps, 4 input features
        #nn.Linear outputs a linear transformation with the inpout tensor as (..., hidden_size) and output tensor of (..., num_classes)
        #self.fc will be applied at the time of prediction since the output should have the final dimension as the number of classes
        #Hidden_size * 2 is required since bidrectional=True doubles the input_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        #Hidden state and cell state (initial values are managed by default)
        out, _ = self.lstm(x)
        #Applying a linear transformation 
        #Out shape: (batch_size, sequence_length, hidden_size)
        output = self.fc(out)
        #output shape: (batch_size, 40, num_classes)
        return output.squeeze(-1)
    

model = LSTM_custom(input_size=4, hidden_size=32, num_layers=1, num_classes=1)

# Hyperparameters
num_epochs = 100
learning_rate = 1e-3

# Class imbalance: compute pos_weight from training data
num_negative = (train_labels[train_masks] == 0).sum().float()
num_positive = (train_labels[train_masks] == 1).sum().float()
#For balancing out classes 
pos_weight = torch.tensor([num_negative / num_positive])

#Binary cross entropy loss. For softmax regression.
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
#Adam_optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Num negative: {num_negative}")
print(f"Num positive: {num_positive}")
print(f"pos_weight: {pos_weight}")

for epoch in range(num_epochs):
    #Puts the model in training mode. Essential for applying dropout or batch normalisation. They are different during training and testing
    model.train()
    #Resetting loss after each epoch to see if it decreases
    total_loss = 0

    for features, labels, masks in train_data_loader:
        #Clears gradients from the previous input batch
        optimizer.zero_grad()
        
        #Cloning the labels so the -1s can be zeroed out
        safe_labels = labels.clone()
        safe_labels[~masks] = 0

        #Inputting the features into the LSTM model -> this is the forward pass
        logits = model(features)
        #Calculating the loss
        loss = criterion(logits, safe_labels.float())
        #Calculating loss only over the mas
        loss = (loss * masks.float()).sum() / masks.sum()

        #Backprop function
        loss.backward()
        #Updateing the weihts
        optimizer.step()
        #Summing up the loss over the batches
        total_loss += loss.item()

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for features, labels, masks in test_data_loader:
        #Computing the foward pass
        logits = model(features)
        preds = (torch.sigmoid(logits) > 0.35).long()

        # Only evaluate on real frames, not padding
        all_preds.append(preds[masks])
        all_true.append(labels[masks])

all_preds = torch.cat(all_preds).numpy()
all_true = torch.cat(all_true).numpy()

from sklearn.metrics import classification_report
print(classification_report(all_true, all_preds, target_names=["non-contact", "contact"]))

print(all_preds)
print(all_true)