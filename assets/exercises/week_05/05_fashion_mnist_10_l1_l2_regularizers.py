import matplotlib.pyplot as plt
import pickle
import os
from utils_fmnist import load_data_full, create_train_model_10

# Daten laden
(x_train, y_train), (x_test, y_test) = load_data_full()

pickle_file = 'model_histories_reg.pkl'

try:
    # Load the history objects from the file
    with open(pickle_file, 'rb') as f:
        histories = pickle.load(f)
    print("Loaded histories from cache.")
except Exception as e:
    print(f"Could not load histories ({e}), training models...")
    import sys
    numepochs = 100
    if len(sys.argv) > 1:
        numepochs = int(sys.argv[1])

    # Modelle trainieren und die Historien speichern
    history_none, _ = create_train_model_10(x_train, y_train, x_test, y_test, hidden_layer_fun='relu',
                                            fit=True, numepochs=numepochs, batch_size=32, regularizer=None)
    history_l1, _ = create_train_model_10(x_train, y_train, x_test, y_test, hidden_layer_fun='relu',
                                          fit=True, numepochs=numepochs, batch_size=32, regularizer='l1')
    history_l2, _ = create_train_model_10(x_train, y_train, x_test, y_test, hidden_layer_fun='relu',
                                          fit=True, numepochs=numepochs, batch_size=32, regularizer='l2')

    histories = {
        'None': history_none.history,
        'l1': history_l1.history,
        'l2': history_l2.history
    }
    # Save the history objects to a file
    with open(pickle_file, 'wb') as f:
        pickle.dump(histories, f)

# Plots erstellen
num_epochs_actual = len(next(iter(histories.values()))['loss'])
epochs = range(1, num_epochs_actual + 1)

# Loss Plot
plt.figure(figsize=(10, 6))
for reg_type in ['None', 'l1', 'l2']:
    plt.plot(epochs, histories[reg_type]['loss'], label=f'Train Loss ({reg_type})')
    plt.plot(epochs, histories[reg_type]['val_loss'], linestyle='--', label=f'Test Loss ({reg_type})')

plt.title('Training and Test Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Accuracy Plot
plt.figure(figsize=(10, 6))
for reg_type in ['None', 'l1', 'l2']:
    plt.plot(epochs, histories[reg_type]['accuracy'], label=f'Train Accuracy ({reg_type})')
    plt.plot(epochs, histories[reg_type]['val_accuracy'], linestyle='--', label=f'Test Accuracy ({reg_type})')

plt.title('Training and Test Accuracy over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
