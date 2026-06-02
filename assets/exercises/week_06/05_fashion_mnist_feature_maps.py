import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np

# Plot-Einstellungen für Präsentationen
plt.rcParams.update({'font.size': 16})

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

# 2. Vorverarbeitung
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test_full.astype('float32') / 255.0
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Validierungsset erstellen
x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 3. Modell erstellen (Gleiches Design wie in 02_fashion_mnist_cnn.ipynb)
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

# 4. Modell kompilieren und kurz trainieren (oder laden, falls vorhanden)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

print("Trainiere Modell für eine Epoche...")
model.fit(x_train, y_train, epochs=1, validation_data=(x_valid, y_valid))

# Modell speichern und wieder laden
model.save("fashion_mnist_cnn.keras")
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# 5. Feature Maps visualisieren
# Ein Bild aus dem Testset auswählen
img_index = 0
input_image = x_test[img_index:img_index+1]

# Klassen-Labels für Fashion MNIST
class_names = ['T-Shirt/Top', 'Hose', 'Pullover', 'Kleid', 'Mantel',
               'Sandale', 'Shirt', 'Sneaker', 'Tasche', 'Stiefelette']

plt.figure(figsize=(5, 5))
plt.imshow(input_image[0, :, :, 0], cmap='gray')
plt.title(f"Eingabebild: {class_names[y_test_full[img_index]]}")
plt.axis('off')
plt.show()

# Ein Modell erstellen, das die Aktivierungen der Layer ausgibt
# Wir nehmen nur die Conv2D Layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Aktivierungen für das Eingabebild berechnen
activations = activation_model.predict(input_image)

# Feature Maps plotten
layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    # Begrenzung auf maximal 64 Feature Maps für die Anzeige
    n_features = min(n_features, 64)
    size = layer_activation.shape[1]
    n_cols = n_features // 8
    
    display_grid = np.zeros((size, n_cols * size))
    
    for col in range(n_cols):
        for row in range(8):
            channel_image = layer_activation[0, :, :, col * 8 + row]
            # Normalisierung für die Visualisierung
            channel_image -= channel_image.mean()
            channel_image /= (channel_image.std() + 1e-5)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[:, col * size : (col + 1) * size] = channel_image # This was wrong in my head, let's fix it

    # Re-do grid logic for better layout
    images_per_row = 8
    n_rows = n_features // images_per_row
    display_grid = np.zeros((n_rows * size, images_per_row * size))
    
    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_activation[0, :, :, row * images_per_row + col]
            channel_image -= channel_image.mean()
            channel_image /= (channel_image.std() + 1e-5)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * size : (row + 1) * size,
                         col * size : (col + 1) * size] = channel_image

    scale = 2.0
    plt.figure(figsize=(scale * images_per_row, scale * n_rows))
    plt.title(f"Feature Maps für Layer: {layer_name}")
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.show()
