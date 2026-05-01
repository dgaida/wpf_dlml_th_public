import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import numpy as np

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

# 2. Vorverarbeitung
# LeNet benötigt 32x32 Eingabebilder. Wir füllen die 28x28 Bilder mit Nullen auf (Padding).
def preprocess_images(images):
    # Padding von 28x28 auf 32x32
    images = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode='constant')
    # Normalisierung
    images = images.astype('float32') / 255.0
    # Kanal-Dimension hinzufügen (32, 32, 1)
    images = images[..., np.newaxis]
    return images

x_train_full = preprocess_images(x_train_full)
x_test = preprocess_images(x_test_full)
y_test = y_test_full

# Validierungsset erstellen (erste 5000 Datenpunkte)
x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 3. Shape des Trainingssets ausgeben
print(f"Shape des Trainingssets: {x_train.shape}")

# 4. Modell erstellen (LeNet-Architektur)
model = Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1), padding='valid'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', padding='valid'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=120, kernel_size=(5, 5), activation='tanh', padding='valid'),
    Flatten(),
    Dense(units=84, activation='tanh'),
    Dense(units=10, activation='softmax')
])

# 5. Modellzusammenfassung ausgeben
print(model.summary())

# 6. Modell kompilieren
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 7. Modell trainieren
# Wir trainieren für 10 Epochen und nutzen das Validierungsset.
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid, y_valid))

# 8. Evaluierung auf dem Testdatensatz
print("\nEvaluierung auf dem Testdatensatz:")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test-Genauigkeit: {test_acc}")

# 9. Vorhersagen für die ersten 3 Datenpunkte des Testsets
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
