import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data()

try:
    # Load the history objects from the file
    with open('model_histories.pkl', 'rb') as f:
        histories = pickle.load(f)

    # Access individual histories
    history_default = histories['default']
    history_linear = histories['linear']
    history_relu = histories['relu']

    model_default = load_model('model_binary_default.h5')
    model_linear = load_model('model_binary_linear.h5')
    model_relu = load_model('model_binary_relu.h5')
except Exception as e:
    # Modelle trainieren und die Historien speichern
    history_default, model_default = create_train_model(x_train, y_train, x_test, y_test, fit=True, numepochs=100)
    history_linear, model_linear = create_train_model(x_train, y_train, x_test, y_test, 'linear', fit=True, numepochs=2)
    history_relu, model_relu = create_train_model(x_train, y_train, x_test, y_test, 'relu', fit=True, numepochs=2)

    # Save the history objects to a file
    with open('model_histories.pkl', 'wb') as f:
        pickle.dump({
            'default': history_default.history,
            'linear': history_linear.history,
            'relu': history_relu.history
        }, f)

    model_default.save('model_binary_default.h5')
    model_linear.save('model_binary_linear.h5')
    model_relu.save('model_binary_relu.h5')

# Trainingsfehler (loss) aus den Historien extrahieren
epochs = range(1, 101)  # 100 Epochen
loss_default = history_default['loss']
loss_linear = history_linear['loss']
loss_relu = history_relu['loss']

try:
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_default, label='Default (no hidden layer)', color='blue')
    plt.plot(epochs, loss_linear, label='Linear activation', color='green')
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

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_default, label='Default (no hidden layer)', color='blue')

plt.title('Training Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.savefig('training_loss_plot_1layer.png', dpi=300, bbox_inches='tight')
