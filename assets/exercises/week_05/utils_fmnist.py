from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def load_data_full():
    # Fashion MNIST-Daten laden
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalisierung der Eingabedaten
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)


# Funktion, um nur die Klassen "Pullover" (2) und "Hemd" (6) zu laden
def load_data():
    # Laden des Fashion MNIST Datensatzes
    (x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

    # Klassen filtern: Pullover (2) und Hemd (6)
    train_filter = np.isin(y_train_full, [2, 6])
    test_filter = np.isin(y_test_full, [2, 6])

    x_train = x_train_full[train_filter]
    y_train = y_train_full[train_filter]
    x_test = x_test_full[test_filter]
    y_test = y_test_full[test_filter]

    # Labels umwandeln: Pullover (2) -> 0, Hemd (6) -> 1
    y_train = np.where(y_train == 2, 0, 1)
    y_test = np.where(y_test == 2, 0, 1)

    # Normalisierung der Pixelwerte (0–255 -> 0–1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)


# Funktion zur Erstellung und zum Training von Modellen mit Rückgabe der Trainingshistorie
def create_train_model_10(x_train, y_train, x_test, y_test, hidden_layer_fun=None, fit=True, numepochs=100,
                          batch_size=32, dropout=False):
    # Modell erstellen
    model = Sequential()
    model.add(Flatten(input_shape=[28, 28]))
    if hidden_layer_fun is not None:
        model.add(Dense(units=300, activation=hidden_layer_fun))
        if dropout:
            model.add(Dropout(0.2))
        model.add(Dense(units=100, activation=hidden_layer_fun))
        if dropout:
            model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    print(model.summary())

    if fit:
        # Modell trainieren
        history = model.fit(x_train, y_train, epochs=numepochs, validation_data=(x_test, y_test),
                            batch_size=batch_size, verbose=1)
    else:
        return None, model

    return history, model
