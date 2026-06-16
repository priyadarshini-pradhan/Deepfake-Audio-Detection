import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("Models/deepfake_cnn.h5")

audio_file = input("Enter audio file path: ")

audio, sr = librosa.load(audio_file, sr=16000)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

mfcc = np.mean(mfcc.T, axis=0)

mfcc = mfcc.reshape(1, 40, 1)

prediction = model.predict(mfcc)

label = np.argmax(prediction)

if label == 0:
    print("\nPrediction: REAL AUDIO")
else:
    print("\nPrediction: FAKE AUDIO")

print("Confidence:", np.max(prediction) * 100, "%")