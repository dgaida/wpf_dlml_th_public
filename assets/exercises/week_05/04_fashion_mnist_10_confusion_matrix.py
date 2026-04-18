import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

from utils_fmnist import *


# Daten laden
(x_train, y_train), (x_test, y_test) = load_data_full()

try:
    # Load the history objects from the file
    with open('model_histories_cm.pkl', 'rb') as f:
        histories = pickle.load(f)

    # Access individual histories
    history_mini = histories['mini']
    # Modelle laden
    model = load_model('model_cm.h5')
except Exception as e:
    # Modelle trainieren und die Historien speichern
    history_mini, model = create_train_model_10(x_train, y_train, x_test, y_test, 'relu', fit=True,
                                                numepochs=600, batch_size=32)

    # Save the history objects to a file
    with open('model_histories_cm.pkl', 'wb') as f:
        pickle.dump({
            'mini': history_mini.history,
        }, f)

    model.save('model_cm.h5')

# Vorhersagen auf den Testdaten
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Klassenindex für jede Vorhersage

# Berechnung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot mit Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Speichern als PNG
plt.show()


# Darstellung mit ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='Blues', xticks_rotation=45)  # Farbkarte und Achsendrehung

# Speichern der Confusion Matrix als Bild
plt.savefig('confusion_matrix_display.png', dpi=300, bbox_inches='tight')  # Speichern in hoher Auflösung
plt.title('Confusion Matrix')
plt.show()

