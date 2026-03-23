from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import Tuple, Optional, Any, Union


def load_data_full() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads and normalizes the full Fashion MNIST dataset.

    Returns:
        A tuple containing the training and test sets as (x_train, y_train), (x_test, y_test).
    """
    # Fashion MNIST-Daten laden
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalisierung der Eingabedaten
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)


def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads and normalizes the Fashion MNIST dataset, filtered for "Pullover" (2) and "Shirt" (6).

    Returns:
        A tuple containing the filtered training and test sets as (x_train, y_train), (x_test, y_test).
    """
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


def create_train_model_10(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_fun: Optional[str] = None,
    fit: bool = True,
    numepochs: int = 100,
    batch_size: int = 32,
    dropout: bool = False,
    regularizer: Optional[str] = None
) -> Tuple[Optional[tf.keras.callbacks.History], tf.keras.Model]:
    """Creates and trains a model for Fashion MNIST with 10 classes.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
        y_test: Test data labels.
        hidden_layer_fun: Activation function for hidden layers.
        fit: Whether to train the model.
        numepochs: Number of training epochs.
        batch_size: Batch size for training.
        dropout: Whether to include dropout layers.
        regularizer: Regularizer type ("l1", "l2", or None).

    Returns:
        A tuple containing the training history and the trained model.
    """
    # Modell erstellen
    model = Sequential()
    model.add(Flatten(input_shape=[28, 28]))

    reg = None
    if regularizer == 'l1':
        reg = regularizers.l1(0.01)
    elif regularizer == 'l2':
        reg = regularizers.l2(0.01)

    if hidden_layer_fun is not None:
        model.add(Dense(units=300, activation=hidden_layer_fun,
                        kernel_regularizer=reg, bias_regularizer=reg))
        if dropout:
            model.add(Dropout(0.2))
        model.add(Dense(units=100, activation=hidden_layer_fun,
                        kernel_regularizer=reg, bias_regularizer=reg))
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
