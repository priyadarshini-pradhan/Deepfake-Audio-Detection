import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load features
X = np.load("Features/X.npy")
y = np.load("Features/y.npy")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# One-hot encoding
y = to_categorical(y, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# CNN Model
model = Sequential()

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=(40, 1)
    )
)

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Training Accuracy', 'Validation Accuracy'])

plt.savefig("Results/accuracy_graph.png")
# Save model
model.save("Models/deepfake_cnn.h5")

print("Model Saved Successfully!")