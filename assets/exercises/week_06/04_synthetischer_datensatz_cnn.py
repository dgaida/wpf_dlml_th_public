import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Reproduzierbarkeit sicherstellen
tf.random.set_seed(42)
np.random.seed(42)

# AGENTS.md: German labels, large font size, high resolution
plt.rcParams.update({'font.size': 16})

# 1. Synthetischen Datensatz erstellen
def create_dataset():
    """Erstellt einen künstlichen Datensatz mit 3 Klassen: Horizontal, Vertikal, Kreuz."""
    images = []
    labels = []
    
    for class_id in range(3):
        for _ in range(10):
            img = np.zeros((24, 24), dtype=np.float32)
            # Leichte Variation der Position (+/- 2 Pixel)
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            
            center_x, center_y = 12 + offset_x, 12 + offset_y
            
            if class_id == 0 or class_id == 2: # Horizontaler Strich (16x4)
                img[max(0, center_y-2):min(24, center_y+2), max(0, center_x-8):min(24, center_x+8)] = 1.0
            
            if class_id == 1 or class_id == 2: # Vertikaler Strich (16x4)
                img[max(0, center_y-8):min(24, center_y+8), max(0, center_x-2):min(24, center_x+2)] = 1.0
                
            images.append(img)
            labels.append(class_id)
            
    # Hinzufügen der Kanal-Dimension und Konvertierung in Numpy-Arrays
    return np.array(images)[..., np.newaxis], np.array(labels)

x_train, y_train = create_dataset()

# 2. CNN Modell definieren
# Anforderungen: 
# - Conv2D: 2 Filter, 3x3, ReLU
# - Conv2D: 1 Filter, 3x3, ReLU
# - Flatten
# - Dense: 3 Neuronen, Softmax
model = keras.Sequential([
    keras.layers.Conv2D(2, (3, 3), activation='relu', input_shape=(24, 24, 1), name='conv_1'),
    keras.layers.Conv2D(1, (3, 3), activation='relu', name='conv_2'),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax', name='output')
], name="Synthetisches_CNN")

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 3. Modell trainieren
print("Training startet...")
history = model.fit(x_train, y_train, epochs=100, verbose=0)
print(f"Training abgeschlossen. Finale Accuracy: {history.history['accuracy'][-1]:.4f}")

# 4. Visualisierung der Kernel
def plot_kernels(model):
    """Visualisiert die Gewichte (Kernel) der Convolutional Layer."""
    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    
    for i, layer in enumerate(conv_layers):
        weights, _ = layer.get_weights()
        # weights shape: (kernel_height, kernel_width, input_channels, output_filters)
        n_inputs = weights.shape[2]
        n_filters = weights.shape[3]
        
        fig, axes = plt.subplots(n_inputs, n_filters, 
                                 figsize=(n_filters * 4, n_inputs * 4), 
                                 squeeze=False, dpi=300)
            
        for f in range(n_filters):
            for inp in range(n_inputs):
                ax = axes[inp, f]
                ax.imshow(weights[:, :, inp, f], cmap='gray')
                ax.set_title(f'Filter {f+1}, In-Kanal {inp+1}')
                ax.axis('off')
        
        plt.suptitle(f"Kernel von {layer.name} (Layer {i+1})")
        plt.tight_layout()
        plt.show()

print("\nVisualisierung der gelernten Kernel:")
plot_kernels(model)

# 5. Visualisierung der Feature Maps
def plot_feature_maps(model, img, title="Feature Maps"):
    """Visualisiert die Aktivierungen (Feature Maps) für ein gegebenes Bild."""
    # Sub-Modell für Zwischenausgaben erstellen
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
    activation_model = keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    
    activations = activation_model.predict(img[np.newaxis, ...], verbose=0)
    
    # Sicherstellen, dass activations eine Liste ist (bei nur einem Layer)
    if not isinstance(activations, list):
        activations = [activations]
        
    for i, activation in enumerate(activations):
        n_filters = activation.shape[-1]
        fig, axes = plt.subplots(1, n_filters, figsize=(n_filters * 4, 4), dpi=300)
        if n_filters == 1:
            axes = [axes]
            
        for j in range(n_filters):
            axes[j].imshow(activation[0, :, :, j], cmap='viridis')
            axes[j].set_title(f'Layer {i+1}, Filter {j+1}')
            axes[j].axis('off')
        
        plt.suptitle(f"{title} - Layer {i+1}")
        plt.tight_layout()
        plt.show()

# Ein Beispielbild pro Klasse visualisieren
klassen_namen = ["Horizontal", "Vertikal", "Kreuz"]
for class_id in range(3):
    print(f"\nFeature Maps für Klasse: {klassen_namen[class_id]}")
    img = x_train[class_id * 10]
    
    # Originalbild zeigen
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Original: {klassen_namen[class_id]}")
    plt.axis('off')
    plt.show()
    
    plot_feature_maps(model, img, title=f"Aktivierungen ({klassen_namen[class_id]})")
