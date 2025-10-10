import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_images_ubyte(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    return images

def load_labels_ubyte(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels

# Cargar algunos ejemplos
root = Path(__file__).parent.parent
dataset_dir = root / "data" / "mnist"

train_images = load_images_ubyte(dataset_dir / "train-images.idx3-ubyte")
train_labels = load_labels_ubyte(dataset_dir / "train-labels.idx1-ubyte")

# Mostrar las primeras 5 imágenes
for i in range(5):
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.show()
