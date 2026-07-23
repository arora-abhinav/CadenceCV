#The purpose of this model is to potentially replace the LSTM for gait cycle classification. This is an SVM
#It reliably worked on strikefoot classification and so I am now expanding to gait cycle classification
import json
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support


with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/strikefoot_keypoints.json", "r") as file:
    strikefoot_kpts = json.load(file)

#non_strikefoot was actually every frame of every video, so it got relabelled to frame_by_frame_keypoint_data
with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/frame_by_frame_keypoint_data.json", "r") as file:
    frame_by_frame_kpts = json.load(file)

#strikefoot_data is where the u frame (toe off) and d frame (strikefoot) live for each leg. Thats what tells
#us where every gait cycle starts and ends
with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/strikefoot_data.json", "r") as file:
    strikefoot_data = json.load(file)

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
frame_by_frame_kpts = flatten_kpts(frame_by_frame_kpts)

strikefoot_df = pd.DataFrame(strikefoot_kpts)
frame_by_frame_df = pd.DataFrame(frame_by_frame_kpts)

#The frame_by_frame video names still carry their file extension (Video10.mp4) while strikefoot_data uses
#the bare name (Video10), so strip the extension otherwise the two never actually join
frame_by_frame_df["video"] = frame_by_frame_df["video"].apply(lambda v: os.path.splitext(v)[0])

#Combining both keypoint files into one pose-per-frame pool (frame_by_frame misses a handful of strikefoot
#frames, strikefoot_kpts fills them back in) and dropping duplicate (video, Frame) rows
keypoint_df = pd.concat([strikefoot_df, frame_by_frame_df], axis=0)
keypoint_df = keypoint_df.drop_duplicates(subset=["video", "Frame"]).reset_index(drop=True)

#Contact is per leg, so the same frame can be a contact frame for one leg and a swing frame for the other.
#Duplicating every frame into an L row and an R row lets each leg carry its own label
left_df = keypoint_df.copy(); left_df["side"] = "L"
right_df = keypoint_df.copy(); right_df["side"] = "R"
keypoint_df = pd.concat([left_df, right_df], axis=0)

#Sorting by frame and side (ascending) exactly like footnet did, so the labelling loop below is deterministic
keypoint_df = keypoint_df.sort_values(["video", "side", "Frame"]).reset_index(drop=True)

#The strikes that give us the gait cycle boundaries. Dropping any strike with no u frame since I cant build
#a cycle without knowing the toe off. Sorting by frame per leg so "previous strike" is just the last row
strikes_df = pd.DataFrame(strikefoot_data)[["video", "side", "frame", "u_frame"]].dropna(subset=["u_frame"])
strikes_df = strikes_df.sort_values(["video", "side", "frame"]).reset_index(drop=True)

#Same labelling loop as footnet: for each strike (the d frame), the frames from this strike's u frame up to
#the d frame are the swing / non contact (0), and the frames from the PREVIOUS same leg strike up to this u
#frame are the stance / contact (1). Same video and same leg of course. Everything else stays None and gets
#dropped since it isnt inside a labelled cycle
keypoint_df["label"] = None
for _, row in strikes_df.iterrows():
    v, side, d, u = row["video"], row["side"], row["frame"], row["u_frame"]

    #non contact: this strike's u frame up to its d frame
    keypoint_df.loc[
        (keypoint_df["video"] == v) &
        (keypoint_df["side"] == side) &
        (keypoint_df["Frame"] >= u) &
        (keypoint_df["Frame"] < d),
        "label"
    ] = 0

    #contact: from the previous same leg strike's d frame up to this strike's u frame
    same_side_prev = strikes_df[
        (strikes_df["video"] == v) &
        (strikes_df["side"] == side) &
        (strikes_df["frame"] < d)
    ]
    if not same_side_prev.empty:
        prev_d = same_side_prev.iloc[-1]["frame"]
        keypoint_df.loc[
            (keypoint_df["video"] == v) &
            (keypoint_df["side"] == side) &
            (keypoint_df["Frame"] >= prev_d) &
            (keypoint_df["Frame"] < u),
            "label"
        ] = 1

#Only keeping the frames that actually landed inside a labelled gait cycle
labelled_df = keypoint_df.dropna(subset=["label"]).reset_index(drop=True)
labelled_df["label"] = labelled_df["label"].astype(int)

kp_cols = ["keypoint " + str(i) for i in range(21)]

#Building the sklearn feature matrix: the 21 [x, y] keypoints flattened to 42, plus a side flag. The side
#flag matters because the pose is whole body, so without it the exact same frame would be a contact example
#for one leg and a non contact example for the other, i.e. identical features with opposite labels
def build_xy(df):
    X = np.array(df[kp_cols].values.tolist()).reshape(len(df), -1)
    side_flag = (df["side"] == "R").astype(int).values.reshape(-1, 1)
    X = np.concatenate([X, side_flag], axis=1)
    y = df["label"].values.astype(int)
    groups = df["video"].values
    return X, y, groups


#degree and coef0 default to 3 and 1 (scikit learn's poly defaults) so I can swap kernel="poly" in freely
def experiment_gait_cycle_svm(C=10, kernel="rbf", gamma="scale", degree=3, coef0=1):
    X, y, groups = build_xy(labelled_df)

    #Grouped by video so every frame of one runner stays on the same side of the split. Per frame data is
    #super autocorrelated (frame t looks almost identical to t+1), so a random split would leak and hand me
    #a fake score. Stratified as well so each fold keeps the contact / non contact balance
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_macro, fold_n_f1, fold_y_f1 = [], [], []
    fold_macro_recall, fold_n_recall, fold_y_recall = [], [], []
    fold_macro_precision, fold_n_precision, fold_y_precision = [], [], []
    fold_macro_support, fold_n_support, fold_y_support = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        #StandardScaler fit on the train fold only (RBF is distance based). Fresh SVM every fold. #A StandardScaler essentially 
        #applies a zscore transformation. The reason the make_pipeline is used is because it ensures that the StandardScaler is 
        #not applied to the testing data, so that the model doesnt memorise new means and averages of the training data but instead
        #uses only the averages and means learnt from the training data
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        precision, recall, f1, support = precision_recall_fscore_support(y_val, preds, labels=[0, 1], zero_division=0)
        macro = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_macro.append(macro); fold_n_f1.append(f1[0]); fold_y_f1.append(f1[1])
        fold_macro_recall.append(np.mean(recall)); fold_n_recall.append(recall[0]); fold_y_recall.append(recall[1])
        fold_macro_precision.append(np.mean(precision)); fold_n_precision.append(precision[0]); fold_y_precision.append(precision[1])
        fold_macro_support.append(np.sum(support)); fold_n_support.append(support[0]); fold_y_support.append(support[1])
        print(f"Fold {fold+1}: macro F1 {macro:.3f}  |  non-contact f1 {f1[0]:.3f}  contact f1 {f1[1]:.3f}")

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
    #N is non contact (label 0), Y is contact (label 1) - same N / Y naming as the other models so it logs the same
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
                "Kernel": kernel,
                "C": C,
                "Gamma": gamma,
                "Degree": degree,
                "Coef0": coef0,
                "Trained Model": clf
                }
