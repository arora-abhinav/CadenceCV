#Using an SVM to detect strikefoot and run corrections on the LSTM's prediction
import json
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score, precision_score
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/strikefoot_keypoints.json", "r") as file:
    strikefoot_kpts = json.load(file)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/data/cleaned up data/frame_by_frame_keypoint_data.json", "r") as file:
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
sampling_num = 1000
non_strikefoot_df = non_strikefoot_df.iloc[:sampling_num]

#Runners in videos are facing 2 different directions: left and right both. This means that the normalised bbox keypoint coordinates are changed
#and therefore duplicating our data to show flipped versions of the strikefoot will help the model understand strikefoot regardless of what direction
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


#Trying a support vector machine with an RBF or polynomial kernel. The RBF kernel lifts the 42-dim
#keypoints into a higher dimensional space where a linear boundary CAN separate classes that arent
#linearly separable in the original space - the hope is a tighter boundary around strikefoot poses,
#especially the tricky +-1 frames that look almost identical to a real strikefoot.
#The degree = 3 and the coef0 = 1 are the SVM's default parameters according to Scikit Learn's documentation.
#The use of an SVM is dependant on the fact that there isn't much data and so SVM is a better choice compared to an MLP
def experiment_svm(C=2, kernel="rbf", gamma="scale", degree=3, coef0=1):
    #Fold on the un-mirrored originals, mirror only the train side of each fold, evaluate on the honest held-out fold.
    #degree and coef0 only matter for the poly kernel and default to 0, so they must be explicitly set for poly
    if kernel == "poly" and (degree == 0 or coef0 == 0):
        raise ValueError("For kernel='poly', degree and coef0 must be explicitly set (they default to 0).")

    def df_to_xy(df):
        #Flatten each row's 21 [x, y] keypoints into one 42-length vector for sklearn
        X = np.array(df.drop(["Label", "video", "Frame"], axis=1).values.tolist()).reshape(len(df), -1)
        y = df["Label"].values.astype(int)
        return X, y

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

        #Mirror ONLY the training fold so a pose and its flip never straddle train/val (no leakage)
        train_aug = pd.concat([train_df, mirror_dfs(train_df)], axis=0)
        X_train, y_train = df_to_xy(train_aug)
        X_val, y_val = df_to_xy(val_df)

        #StandardScaler because the RBF kernel is distance based, so features should be on the same scale.
        #It fits on the TRAIN fold only then transforms val, so no validation stats leak in. Fresh SVM each fold.
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        true = y_val

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
                "Kernel": kernel,
                "C": C,
                "Gamma": gamma,
                "Degree": degree,
                "Coef0": coef0,
                "Sampling Num": sampling_num,
                "Trained Model": clf
                }
