#A Script dedicated to loading the testing and training data
import json 
import pandas as pd

json_path = "/Users/abhinavarora/Desktop/CadenceCV/Json Data/strikefoot_data.json"

with open(json_path, "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
#Split into training and testing

#These specific videos comprise 20% of the strikes
testing_video_strings = set(["Video9", "instavid_13", "instavid_15", "Video3"])
training_video_strings = [frame["video"] for frame in data if frame["video"] not in testing_video_strings]

testing_df = df.loc[df["video"].isin(testing_video_strings)].reset_index()
training_df = df.loc[df["video"].isin(training_video_strings)].reset_index()