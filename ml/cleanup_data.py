#The purpose of this script is to clean up some data from strikefoot and non_strikefoot frames
#Some videos aren't fully sideview, so their keypoints come out a bit strange and mess up the data. This
#drops those videos (an explicit list + every .MOV video) out of the data jsons and writes cleaned copies
#into a new folder, leaving the originals untouched
import json
import os

DATA_DIR = "/Users/abhinavarora/Desktop/CadenceCV/ml/data"
VIDEOS_DIR = "/Users/abhinavarora/Desktop/CadenceCV/Videos"
OUT_DIR = os.path.join(DATA_DIR, "cleaned up data")

#Videos I explicitly want gone
explicit_removals = {"instavid_6", "instavid_7", "instavid_9", "instavid_15", "instavid_17"}

#Every .MOV video (they aren't fully sideview). Pulled straight from the Videos folder so the list stays
#correct if more get added. Extensions stripped so it matches whatever name the jsons happen to store
mov_removals = {os.path.splitext(f)[0] for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mov")}

removed_videos = explicit_removals | mov_removals
print("Removing videos:", sorted(removed_videos))

#Some jsons store the extension (Video9.MOV), others store the bare name (Video9), so strip it before comparing
def base_name(v):
    return os.path.splitext(v)[0]

os.makedirs(OUT_DIR, exist_ok=True)

#Cleaning every json in the data folder that is a list of per-video entries
for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except (json.JSONDecodeError, ValueError):
        print(f"skipping {filename} (empty / not valid json)")
        continue

    #Only touching files that are a list of dicts with a video field
    if not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "video" in data[0]):
        print(f"skipping {filename} (no per-video entries)")
        continue

    cleaned = [entry for entry in data if base_name(entry["video"]) not in removed_videos]

    with open(os.path.join(OUT_DIR, filename), "w") as file:
        json.dump(cleaned, file, indent=4)

    print(f"{filename}: {len(data)} -> {len(cleaned)} entries (dropped {len(data) - len(cleaned)})")

print(f"\nCleaned files written to: {OUT_DIR}")
