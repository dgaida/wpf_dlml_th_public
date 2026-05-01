import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

# 2. Vorverarbeitung
# Normalisierung und Hinzufügen der Kanal-Dimension
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test_full.astype('float32') / 255.0
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]
y_test = y_test_full

# Validierungsset erstellen (erste 5000 Datenpunkte)
x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

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

# 6. Modell trainieren
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid, y_valid))

# 7. Evaluierung auf dem Testdatensatz
print("\nEvaluierung auf dem Testdatensatz:")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test-Genauigkeit: {test_acc}")

# 8. Vorhersagen für die ersten 3 Datenpunkte des Testsets
x_new = x_test[:3]

# predict aufrufen
y_proba = model.predict(x_new)
print("\nVorhersagewahrscheinlichkeiten (predict):")
print(y_proba.round(2))

# Vorhergesagte Klassen bestimmen
y_pred = np.argmax(model.predict(x_new), axis=-1)
print("\nVorhergesagte Klassen:")
print(y_pred)
print("Tatsächliche Klassen:")
print(y_test[:3])
