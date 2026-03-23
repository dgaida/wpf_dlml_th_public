from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np  # Falls erforderlich

import matplotlib.pyplot as plt


numepochs = 10


# Dummy-Daten (XOR-Problem)
def load_data():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])
    return (x_train, y_train), (x_train, y_train)


(x_train, y_train), (x_test, y_test) = load_data()

# Model
model = Sequential()
model.add(Dense(units=1, activation='sigmoid', input_shape=(2,)))

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=2.5), metrics=['accuracy'])

# Training
history = model.fit(x_train, y_train, epochs=numepochs, verbose=0)

# Vorhersage
classes = (model.predict(x_test) > 0.5).astype("int32")
print("Predictions:", classes)

# OPTIONAL
# Plot des Trainingsloss über Anzahl der Epochen

epochs = range(1, numepochs + 1)  # 100 Epochen
loss = history.history['loss']

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, color='blue')

plt.title('Training Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.grid(True)

plt.savefig('training_loss_plot_xor_1neuron.png', dpi=300, bbox_inches='tight')
