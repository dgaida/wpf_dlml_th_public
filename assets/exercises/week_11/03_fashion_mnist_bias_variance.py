import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np
import os

# 1. Fashion MNIST Datensatz laden
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# 2. Vorverarbeitung
# Normalisierung und Hinzufügen der Kanal-Dimension
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Aufteilen in Trainings- und Validierungsdaten
x_train, x_valid = x_train_full[:55000], x_train_full[55000:]
y_train, y_valid = y_train_full[:55000], y_train_full[55000:]

# 3. Modell erstellen oder laden
model_path = "fashion_mnist_bias_variance.keras"

if os.path.exists(model_path):
    print("Modell wird geladen...")
    model = keras.models.load_model(model_path)
else:
    print("Neues Modell wird erstellt...")
    tf.random.set_seed(42)

    DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                            activation="relu", kernel_initializer="he_normal")

    model = tf.keras.Sequential([
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPool2D(),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
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

    # Modell kompilieren
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])

# 4. Modell trainieren
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid, y_valid))

# 5. Modell speichern
model.save(model_path)

# 6. Trainings- und Validierungsverlust plotten
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Trainingsfehler')
plt.plot(history.history['val_loss'], label='Validierungsfehler')
plt.title('Trainings- und Validierungsfehler über die Epochen')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.grid(True)
plt.show()

# 7. Bias-Variance Decomposition

# Evaluierung
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, val_acc = model.evaluate(x_valid, y_valid, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)

irreducible_error = 0.001 # 0,1 %
train_error = 1 - train_acc
val_error = 1 - val_acc
test_error = 1 - test_acc

avoidable_bias = max(0, train_error - irreducible_error)
variance = max(0, val_error - train_error)
val_set_overfitting = max(0, test_error - val_error)

# Daten für den Barplot
labels = ['Irreducible error', 'Avoidable bias', 'Train error', 'Variance', 'Val error', 'Val set overfitting', 'Test error']
values = [irreducible_error, avoidable_bias, train_error, variance, val_error, val_set_overfitting, test_error]
colors = ['#4A70BB', '#ED7D31', '#00B050', '#ED7D31', '#FF0000', '#ED7D31', '#7030A0']

fig, ax = plt.subplots(figsize=(12, 7))

level_indices = [0, 2, 4, 6]
diff_indices = [1, 3, 5]

# Haupt-Balken
for i in level_indices:
    ax.bar(i, values[i] * 100, color=colors[i])
    ax.text(i, values[i] * 100 + 0.1, f"{values[i]*100:.2f}%", ha='center', fontweight='bold')

# Differenz-Balken (Floating)
bottoms = [values[0], values[2], values[4]]
for i, idx in enumerate(diff_indices):
    ax.bar(idx, values[idx] * 100, bottom=bottoms[i] * 100, color=colors[idx])
    ax.text(idx, (bottoms[i] + values[idx]) * 100 + 0.1, f"{values[idx]*100:.2f}%", ha='center')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_title("Breakdown of test error by source")
ax.set_ylabel("Error Rate (%)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

### Aufgabe für Studierende
# reduzieren Sie den Fehler, den Sie laut Vorlesung zuerst verkleinern sollten mit den laut Vorlesung vielversprechendsten Methoden.
