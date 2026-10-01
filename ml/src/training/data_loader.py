#A Script dedicated to loading the testing and training data
import json 
import pandas as pd

#Training on the UNCORRECTED data on purpose. A model retrained on the side corrected version
#(ml/extract_side_corrected_data.py -> normalised_strikefoot_data_side_corrected.json) scored better frame by frame
#but its toe offs got worse on unseen videos (held out 0.85 -> 0.76, Video18 0.93 -> 0.78 F1), even after tuning
#the threshold. The side correction is still used at INFERENCE, the uncorrected trained model did best with it
json_path = "/Users/abhinavarora/Desktop/CadenceCV/ml/data/normalised_strikefoot_data.json"

with open(json_path, "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
#Split into training and testing

#These specific videos comprise 20% of the strikes
testing_video_strings = set(["Video9", "instavid_13", "instavid_15", "Video3"])
training_video_strings = [frame["video"] for frame in data if frame["video"] not in testing_video_strings]

testing_df = df.loc[df["video"].isin(testing_video_strings)].reset_index()
training_df = df.loc[df["video"].isin(training_video_strings)].reset_index()