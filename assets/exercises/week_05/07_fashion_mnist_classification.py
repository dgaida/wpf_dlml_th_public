import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 2. Validierungsset erstellen (erste 5000 Datenpunkte) und Normalisierung
# Das Validierungsset wird zur Überwachung der Genauigkeit während des Trainings verwendet.
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test / 255.0

# 3. Shape und Dtype des Trainingssets ausgeben
print(f"Shape des Trainingssets: {x_train.shape}")
print(f"Dtype des Trainingssets: {x_train.dtype}")

# 4. Modell erstellen (wie im bereitgestellten Bild)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# 5. Modellzusammenfassung ausgeben
print(model.summary())

# 6. Modell kompilieren (mit Adam-Optimizer für bessere Performance)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 7. Modell trainieren
# Während des Trainings wird die Genauigkeit des Validierungssets (val_accuracy) angezeigt.
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

# predict_classes aufrufen
# Hinweis: predict_classes ist in neueren TensorFlow-Versionen entfernt.
# Wir verwenden np.argmax als direkten Ersatz.
try:
    y_pred = model.predict_classes(x_new)
    print("\nVorhergesagte Klassen (predict_classes):")
except AttributeError:
    y_pred = np.argmax(model.predict(x_new), axis=-1)
    print("\nVorhergesagte Klassen (Ersatz für predict_classes):")

print(y_pred)
print("Tatsächliche Klassen:")
print(y_test[:3])
