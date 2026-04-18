import pickle
from tensorflow.keras.models import load_model

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data_full()

try:
    # Load the history objects from the file
    with open('model_histories_drop.pkl', 'rb') as f:
        histories = pickle.load(f)

    # Access individual histories
    history_mini = histories['drop']
    # Modelle laden
    model_relu = load_model('model_drop.h5')
except Exception as e:
    # Modelle trainieren und die Historien speichern
    history_drop, model = create_train_model_10(x_train, y_train, x_test, y_test, 'relu', fit=True,
                                                numepochs=600, batch_size=32, dropout=True)

    # Save the history objects to a file
    with open('model_histories_drop.pkl', 'wb') as f:
        pickle.dump({
            'drop': history_drop.history,
        }, f)

    model.save('model_drop.h5')
