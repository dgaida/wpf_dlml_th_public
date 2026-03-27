import matplotlib.pyplot as plt
import pickle

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data()

try:
    # Load the history objects from the file
    with open('model_histories_batch.pkl', 'rb') as f:
        histories = pickle.load(f)

    # Access individual histories
    history_vanilla = histories['vanilla']
    history_sgd = histories['sgd']
    history_mini = histories['mini']
except Exception as e:
    # Modelle trainieren und die Historien speichern
    history_vanilla, model_vanilla = create_train_model(x_train, y_train, x_test, y_test, 'relu', fit=True,
                                                        numepochs=1000, batch_size=x_train.shape[0])
    history_sgd, model_sgd = create_train_model(x_train, y_train, x_test, y_test, 'relu', fit=True,
                                                numepochs=200, batch_size=1)
    history_mini, model_mini = create_train_model(x_train, y_train, x_test, y_test, 'relu', fit=True,
                                                  numepochs=600, batch_size=32)

    # Save the history objects to a file
    with open('model_histories_batch.pkl', 'wb') as f:
        pickle.dump({
            'vanilla': history_vanilla.history,
            'sgd': history_sgd.history,
            'mini': history_mini.history
        }, f)

# Trainingsfehler (loss) aus den Historien extrahieren
epochs = range(1, 101)  # 100 Epochen
loss_vanilla = history_vanilla.history['loss']
loss_sgd = history_sgd.history['loss']
loss_mini = history_mini.history['loss']

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_vanilla)), loss_vanilla, label='Vanilla', color='blue')
plt.plot(range(1, len(loss_sgd)), loss_sgd, label='Stochastic', color='green')
plt.plot(range(1, len(loss_mini)), loss_mini, label='Mini-Batch', color='red')

plt.title('Training Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
# plt.show()

plt.savefig('training_loss_plot_batchsize.png', dpi=300, bbox_inches='tight')
