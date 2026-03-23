import pickle
from tensorflow.keras.models import load_model

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data_full()

# Modell erstellen
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

print(model.summary())

# Modell trainieren
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),
                    batch_size=32, verbose=1)

# Vorhersagen auf den Testdaten
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Klassenindex für jede Vorhersage

print(y_test[0, :])
print(y_pred_classes[0, :])
