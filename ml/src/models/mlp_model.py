#Using an MLP to detect strikefoot and run corrections on the LSTM's prediction
import json
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from sklearn.metrics import classification_report
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score, precision_score
import numpy as np


with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/strikefoot_keypoints.json", "r") as file:
    strikefoot_kpts = json.load(file)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/non_strikefoot_keypoints.json", "r") as file:
    non_strikefoot_kpts = json.load(file)

#Restructuring the keypoints to flatten out the dict entries in the json file
def flatten_kpts(kpts):
    res = []
    for frame in kpts:
        d = {}
        d["video"] = frame["video"]
        for (index, k) in enumerate(frame["Keypoints"]):
            d["keypoint " + str(index)] = k
        d["Frame"] = frame["Frame"]
        res.append(d)
    
    return res

strikefoot_kpts = flatten_kpts(strikefoot_kpts)
non_strikefoot_kpts = flatten_kpts(non_strikefoot_kpts)

print(strikefoot_kpts)

strikefoot_df = pd.DataFrame(strikefoot_kpts)
non_strikefoot_df = pd.DataFrame(non_strikefoot_kpts)

non_strikefoot_labels = pd.Series([0] * len(non_strikefoot_df), name="Label")
strikefoot_labels = pd.Series([1] * len(strikefoot_df), name="Label")
strikefoot_df = pd.concat([strikefoot_df,strikefoot_labels], axis=1)
non_strikefoot_df = pd.concat([non_strikefoot_df, non_strikefoot_labels], axis=1)

non_strikefoot_df = non_strikefoot_df.dropna(axis=0, how="any").reset_index(drop=True)
strikefoot_df = strikefoot_df.dropna(axis=0, how="any").reset_index(drop=True)

#Dropping the frames of non strikefoot df that are in strikefoot df provided they have the same video
strike_pairs = set(zip(strikefoot_df["video"], strikefoot_df["Frame"]))
mask = ~non_strikefoot_df.apply(lambda r: (r["video"], r["Frame"]) in strike_pairs, axis=1)
non_strikefoot_df = non_strikefoot_df[mask]
non_strikefoot_df = non_strikefoot_df.reset_index(drop=True)
strikefoot_df = strikefoot_df.reset_index(drop=True)

#Sorting by frame and side (but for the same videos)
def sort_by_frame_and_side(df:pd.DataFrame):
    vids = df["video"].unique()
    dfs = []
    for v in vids:
        d:pd.Series = df[df["video"] == v]
        d = d.sort_values(["Frame"], axis=0)
        dfs.append(d)
    
    res = pd.DataFrame(pd.concat(dfs, axis=0)).reset_index(drop=True)
    return res

non_strikefoot_df = sort_by_frame_and_side(non_strikefoot_df)
strikefoot_df = sort_by_frame_and_side(strikefoot_df)

#Shuffling the nonstrikefoot df first and then keeping only 1000 examples to balance out the training data
non_strikefoot_df = non_strikefoot_df.sample(frac=1)
non_strikefoot_df = non_strikefoot_df.iloc[:1000]

#Runners in videos are facing 2 different directions: left and right both. This means that the normalised bbox keypoint coordinates are changed
#and therefore duplicating our data to show flipped versions of the strikefoot will help the MLP understand strikefoot regardless of what direction
#a label is facing (the strikefoot/not a strikefoot label is invariant of direction). 2 things will be required here: flip idx and changing each x coordinate to 1 - x coordinate
flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19]
def mirror_dfs(df:pd.DataFrame):
    df = df.copy()
    #Due to there being 21 keypoints
    for i in range(21):
        df["keypoint " + str(i)] = df["keypoint " + str(i)].apply(lambda k: [1 - k[0], k[1]])

    #Now, swapping the columns based on flip idx:
    cols = {i: df["keypoint " + str(i)].copy() for i in range(21)}
    for i in range(21):
        df["keypoint " + str(i)] = cols[flip_idx[i]]

    return df

flipped_strikefoot_df = mirror_dfs(strikefoot_df)
flipped_non_strikefoot_df = mirror_dfs(non_strikefoot_df)

#Combinig the 4 dataframes to produce my train_test split:
total_df = pd.concat([flipped_strikefoot_df, flipped_non_strikefoot_df, strikefoot_df, non_strikefoot_df], axis=0)
total_df = total_df.sample(frac=1).reset_index(drop=True)

#Now, doing a random 80-20 split to the total df
training_df = total_df.iloc[:int(0.8 * total_df.shape[0])]
testing_df = total_df.iloc[int(0.8 * total_df.shape[0]):, ]

training_labels = training_df["Label"]
testing_labels = testing_df["Label"]

training_featues = training_df.drop(["Label", "video", "Frame"], axis=1)
testing_features = testing_df.drop(["Label", "video", "Frame"], axis=1)

#Converting to tensors:
#BCEWithLogitsLoss compares each output logit against a 0.0/1.0 target, so the labels have to be floats
#here (not the long that torch defaults to from int labels), otherwise the loss throws a dtype error
training_labels = torch.tensor(training_labels.values.tolist(), dtype=torch.float32)
testing_labels = torch.tensor(testing_labels.values.tolist(), dtype=torch.float32)
training_featues = torch.tensor(training_featues.values.tolist())
testing_features = torch.tensor(testing_features.values.tolist())

class MLPDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)
    
training_dataset = MLPDataset(training_featues, training_labels)
testing_dataset = MLPDataset(testing_features, testing_labels)

training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

#The shape of each batch is now (32, 21, 2)

class PoseMLP(nn.Module):
    def __init__(self, dropout = 0.2):
        super(PoseMLP, self).__init__()
        self.dropout = dropout
        #2 rows, 21 keypoints, 21 outputs (just random idk)
        self.fc1 = nn.Linear(2 * 21, 64)
        self.fc1_dropout = nn.Dropout(self.dropout)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        #Propagation into the second layer, 1 output strikefoot or not
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x:torch.Tensor):
        #Features are in the tensor (32, 21, 2). Flattening to (32, 42)
        x = torch.flatten(x, start_dim=1)
        #Passing through a ReLU activation instead of Sigmoid to prevent the vanishing gradient problem
        x = self.relu(self.fc1(x))
        #Applying dropout
        x = self.fc1_dropout(x)
        #Removes the redundant 1 dimension that is required for bce
        x = self.fc2(x).squeeze(-1)
        return x

model = PoseMLP()
#The weight decay > 0 adds L2 Rwgularisation
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#BCEWithLogitsLoss = sigmoid + binary cross entropy fused into one (numerically stable), so we keep
#raw logits in the model and let the loss apply the sigmoid itself. It wants one logit per sample
#We have way more non-strikefoot than strikefoot, so without help the model just learns to say 0.
#pos_weight upweights the rare positive class in the loss: pos_weight = num negatives / num positives
num_pos = (training_labels == 1).sum()
num_neg = (training_labels == 0).sum()
pos_weight = num_neg / num_pos
print(pos_weight)
criterion = nn.BCEWithLogitsLoss()

#K-fold cross validation so the macro F1 isnt just luck from one random 80/20 split. It trains k
#separate models on different folds and averages, giving a mean +- std you can actually compare across
#experiments. We fold on the ORIGINAL (un-mirrored) rows and only mirror the TRAIN side of each fold:
#if a pose and its mirror landed in different folds we'd be validating on a near-duplicate of a training
#example (leakage) and the score would look better than it really is.
def df_to_tensors(df):
    feats = torch.tensor(df.drop(["Label", "video", "Frame"], axis=1).values.tolist())
    labels = torch.tensor(df["Label"].values.tolist(), dtype=torch.float32)
    return feats, labels

#Wrapping the code in an experiment function to track hyperparameters as well as metrics across different runs
def experiment():
    #Seeded so the folds (and therefore the comparison) are reproducible run to run
    torch.manual_seed(42)
    epochs = 100
    lr = 1e-4
    #Weight decay allows L2 Regularization with the AdamW optimizer
    weight_decay = 1e-5
    optimzer_name = "AdamW"
    batch_size = 32
    #The number of neurons in each layer
    model_architecture = [42, 64, 1]

    #The originals = strikefoot + non_strikefoot AFTER dedup/subsample but BEFORE any mirroring
    originals = pd.concat([strikefoot_df, non_strikefoot_df], axis=0).reset_index(drop=True)
    y_all = originals["Label"].values

    #shuffle + stratify so every fold keeps the same strikefoot / non-strikefoot balance
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_macro, fold_n_f1, fold_y_f1 = [], [], []
    fold_macro_recall, fold_n_recall, fold_y_recall = [], [], []
    fold_macro_precision, fold_n_precision, fold_y_precision = [], [], []
    fold_macro_support, fold_n_support, fold_y_support = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(originals, y_all)):
        train_df = originals.iloc[train_idx]
        val_df = originals.iloc[val_idx]

        #Mirror ONLY the training fold, then combine with its originals. Val stays un-mirrored so the score is honest
        train_aug = pd.concat([train_df, mirror_dfs(train_df)], axis=0)
        X_train, y_train = df_to_tensors(train_aug)
        X_val, y_val = df_to_tensors(val_df)
        train_dl = DataLoader(MLPDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        #Fresh model + optimizer every fold so folds dont share learned weights
        model = PoseMLP()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(epochs):
            for features, labels in train_dl:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), labels.float())
                loss.backward()
                optimizer.step()

        #Evaluate on the held-out (un-mirrored) fold
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(X_val)) > 0.45).long().numpy()
        true = y_val.long().numpy()

        precision, recall, f1, support = precision_recall_fscore_support(true, preds, labels=[0, 1], zero_division=0)
        macro = f1_score(true, preds, average="macro", zero_division=0)
        fold_macro.append(macro); fold_n_f1.append(f1[0]); fold_y_f1.append(f1[1])
        fold_macro_recall.append(np.mean(recall)); fold_n_recall.append(recall[0]); fold_y_recall.append(recall[1])
        fold_macro_precision.append(np.mean(precision)); fold_n_precision.append(precision[0]); fold_y_precision.append(precision[1])
        fold_macro_support.append(np.sum(support)); fold_n_support.append(support[0]); fold_y_support.append(support[1])
        print(f"Fold {fold+1}: macro F1 {macro:.3f}  |  N f1 {f1[0]:.3f}  Y f1 {f1[1]:.3f}")

    macro_f1 = np.mean(fold_macro)
    n_f1 = np.mean(fold_n_f1)
    y_f1 = np.mean(fold_y_f1)

    macro_recall = np.mean(fold_macro_recall)
    n_recall = np.mean(fold_n_recall)
    y_recall = np.mean(fold_y_recall)

    macro_precision = np.mean(fold_macro_precision)
    n_precision = np.mean(fold_n_precision)
    y_precision = np.mean(fold_y_precision)

    macro_support = np.mean(fold_macro_support)
    n_support = np.mean(fold_n_support)
    y_support = np.mean(fold_y_support)

    print(f"Macro F1: {macro_f1:.3f} +/- {np.std(fold_macro):.3f}")
    print(f"N f1 : {n_f1:.3f} +/- {np.std(fold_n_f1):.3f}")
    print(f"Y f1 : {y_f1:.3f} +/- {np.std(fold_y_f1):.3f}")

    return {"Macro F1": macro_f1,
                "N F1": n_f1,
                "Y F1": y_f1,
                "Macro Recall": macro_recall,
                "N Recall": n_recall,
                "Y Recall": y_recall,
                "Macro Precision": macro_precision,
                "N Precision": n_precision,
                "Y Precision": y_precision,
                "Macro Support": macro_support,
                "N Support": n_support,
                "Y Support": y_support,
                "Epochs": epochs,
                "Batch_size": batch_size,
                "Model Architecture": model_architecture,
                "Weight Decay": weight_decay,
                "Learning Rate": lr
                }
