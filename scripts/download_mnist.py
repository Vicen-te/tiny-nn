import kagglehub
from pathlib import Path

# Directory where the MNIST dataset will be saved
root = Path(__file__).parent.parent
save_path = root / "data" / "minst"
save_path.mkdir(exist_ok=True)

# Download the MNIST dataset from Kaggle
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset
path = kagglehub.dataset_download("hojjatk/mnist-dataset", path=save_path)

print("Dataset files saved at:", path)