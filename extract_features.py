import os
import random
import librosa
import numpy as np
from tqdm import tqdm

protocol_file = r"DataSet\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
audio_folder = r"DataSet\LA\LA\ASVspoof2019_LA_train\flac"

X = []
y = []

with open(protocol_file, "r") as f:
    lines = f.readlines()

bonafide_lines = []
spoof_lines = []

for line in lines:
    if "bonafide" in line:
        bonafide_lines.append(line)
    else:
        spoof_lines.append(line)

random.shuffle(bonafide_lines)
random.shuffle(spoof_lines)

lines = bonafide_lines[:500] + spoof_lines[:500]

random.shuffle(lines)

for line in tqdm(lines):

    parts = line.strip().split()

    file_id = parts[1]
    label = parts[-1]

    audio_path = os.path.join(audio_folder, file_id + ".flac")

    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        X.append(mfcc)

        if label == "bonafide":
            y.append(0)
        else:
            y.append(1)

    except:
        pass

print("Real:", y.count(0))
print("Fake:", y.count(1))

X = np.array(X)
y = np.array(y)

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)

os.makedirs("Features", exist_ok=True)

np.save("Features/X.npy", X)
np.save("Features/y.npy", y)

print("Features Saved Successfully!")