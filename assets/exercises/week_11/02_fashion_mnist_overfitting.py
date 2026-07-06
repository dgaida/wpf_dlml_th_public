import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 2. Vorverarbeitung
# Normalisierung und Hinzufügen der Kanal-Dimension
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Ein Batch für das Training (32 Bilder), der Rest für die Validierung
# Dies dient dazu zu sehen, ob das Modell komplex genug ist, um 
# diesen kleinen Datensatz perfekt zu memorieren (Overfitting).
x_train_batch = x_train_full[:32]
y_train_batch = y_train_full[:32]
x_valid = x_train_full[32:]
y_valid = y_train_full[32:]

# 3. Modell erstellen (Modernes CNN mit He-Initialisierung)
tf.random.set_seed(42)

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

# 4. Modellzusammenfassung ausgeben
print(model.summary())

# 5. Modell kompilieren
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

# 6. Modell trainieren mit nur einem Batch
# Wir trainieren für viele Epochen, um sicherzustellen, dass der Trainingsfehler auf 0 sinkt.
history = model.fit(x_train_batch, y_train_batch, epochs=100,
                    batch_size=32,
                    validation_data=(x_valid, y_valid))

# 7. Trainings- und Validierungsverlust plotten
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Trainingsfehler')
plt.plot(history.history['val_loss'], label='Validierungsfehler')
plt.title('Trainings- und Validierungsfehler über die Epochen')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.grid(True)
plt.show()

# 8. Evaluierung auf dem Testdatensatz
print("\nEvaluierung auf dem Testdatensatz:")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test-Genauigkeit: {test_acc}")
