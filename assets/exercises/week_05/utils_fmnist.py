from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import Tuple, Optional, Any, Union, List


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


def create_train_model(
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
    """Creates and trains a model for Fashion MNIST binary classification (Pullover vs. Shirt).

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
        y_test: Test data labels.
        hidden_layer_fun: Activation function for the hidden layer.
        fit: Whether to train the model.
        numepochs: Number of training epochs.
        batch_size: Batch size for training.
        dropout: Whether to include a dropout layer.
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
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    print(model.summary())

    if fit:
        # Modell trainieren
        history = model.fit(x_train, y_train, epochs=numepochs, validation_data=(x_test, y_test),
                            batch_size=batch_size, verbose=1)
    else:
        return None, model

    return history, model


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


class GradientLogger(Callback):
    """Callback to log the mean absolute gradients of each trainable layer at the end of each epoch.

    Attributes:
        x_train: Training data features as a tensor.
        y_train: Training data labels as a tensor.
        gradients_history: List to store mean absolute gradients per layer for each epoch.
        layer_names: List of names of layers with trainable weights.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        super(GradientLogger, self).__init__()
        self.x_train = tf.convert_to_tensor(x_train)  # Konvertiere x_train in einen Tensor
        self.y_train = tf.convert_to_tensor(y_train)  # Konvertiere y_train in einen Tensor
        self.gradients_history: List[List[float]] = []  # Speichert die Gradienten pro Epoche
        self.layer_names: List[str] = []

    def on_train_begin(self, logs: Optional[dict] = None):
        self.layer_names = [layer.name for layer in self.model.layers if len(layer.trainable_weights) > 0]

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_train, training=True)
            loss = self.model.compiled_loss(self.y_train, predictions, regularization_losses=self.model.losses)

        trainable_vars = self.model.trainable_weights
        grads = tape.gradient(loss, trainable_vars)

        # Berechnung der mittleren Gradienten pro Schicht (Mittelwert aus Kernel und Bias)
        layer_grads = []
        grad_idx = 0
        for layer in self.model.layers:
            if len(layer.trainable_weights) > 0:
                num_weights = len(layer.trainable_weights)
                current_layer_grads = grads[grad_idx:grad_idx + num_weights]
                # Mittelwert der absoluten Gradienten für die gesamte Schicht berechnen
                mean_abs = np.mean([tf.reduce_mean(tf.abs(g)).numpy() for g in current_layer_grads if g is not None])
                layer_grads.append(float(mean_abs))
                grad_idx += num_weights

        self.gradients_history.append(layer_grads)


def create_train_model_big(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_fun: Optional[str] = None,
    fit: bool = True,
    numepochs: int = 100,
    batch_size: int = 32
) -> Tuple[Optional[tf.keras.callbacks.History], tf.keras.Model, Optional[np.ndarray], Optional[GradientLogger]]:
    """Creates and trains a deep network for demonstrating vanishing gradients.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
        y_test: Test data labels.
        hidden_layer_fun: Activation function for the hidden layers.
        fit: Whether to train the model.
        numepochs: Number of training epochs.
        batch_size: Batch size for training.

    Returns:
        A tuple containing the training history, the model, the gradients array, and the logger instance.
    """
    # Modell erstellen (tiefes Netzwerk zur Demonstration von Vanishing Gradients)
    model = Sequential()
    model.add(Flatten(input_shape=[28, 28]))
    # 15 versteckte Schichten
    for _ in range(15):
        model.add(Dense(units=20, activation=hidden_layer_fun, kernel_initializer='glorot_uniform'))
    model.add(Dense(units=10, activation='softmax', kernel_initializer='glorot_uniform'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

    print(model.summary())

    if fit:
        # Gradienten für ReLU
        logger = GradientLogger(x_train, y_train)

        # Modell trainieren
        history = model.fit(x_train, y_train, epochs=numepochs, validation_data=(x_test, y_test),
                            batch_size=batch_size, callbacks=[logger], verbose=1)

        # Konvertieren der Listen in numpy Arrays
        gradients = np.array(logger.gradients_history)
        return history, model, gradients, logger
    else:
        return None, model, None, None
