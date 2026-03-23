import matplotlib.pyplot as plt

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

print(model.summary())

# Modell trainieren
history_relu = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),
                         batch_size=32, verbose=1)

# Trainingsfehler (loss) aus den Historien extrahieren
epochs = range(1, 101)  # 100 Epochen
# loss_default = history_default['loss']
# loss_linear = history_linear['loss']
loss_relu = history_relu['loss']

try:
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    # plt.plot(epochs, loss_default, label='Default (no hidden layer)', color='blue')
    # plt.plot(epochs, loss_linear, label='Linear activation', color='green')
    plt.plot(epochs, loss_relu, label='ReLU activation', color='red')

    plt.title('Training Loss over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    # plt.show()

    plt.savefig('training_loss_plot_layer.png', dpi=300, bbox_inches='tight')
except Exception as e:
    print(e)
