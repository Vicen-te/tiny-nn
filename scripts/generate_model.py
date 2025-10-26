import json, random, sys
from pathlib import Path
import logging

def gen(in_sz=128, hidden=64, out=10, seed=123):
    """Generate a simple 2-layer fully connected neural network model with random weights and biases."""
    random.seed(seed)
    layers = []

    # First fully connected layer
    w1 = [[random.uniform(-0.5,0.5) for _ in range(in_sz)] for _ in range(hidden)]
    b1 = [random.uniform(-0.1,0.1) for _ in range(hidden)]
    layers.append({
        "name": "fc1",
        "in": in_sz,
        "out": hidden,
        "W": w1,         # 2D array [out][in]
        "b": b1,         # bias vector
        "activation": "relu"
    })

    # Second fully connected layer
    w2 = [[random.uniform(-0.5,0.5) for _ in range(hidden)] for _ in range(out)]
    b2 = [random.uniform(-0.1,0.1) for _ in range(out)]
    layers.append({
        "name": "fc2",
        "in": hidden,
        "out": out,
        "W": w2,         # 2D array
        "b": b2,
        "activation": "softmax"
    })

    return {"input_size": in_sz, "output_size": out, "layers": layers}


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    models_dir = root / "data" / "models"
    models_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
    logger = logging.getLogger()

    in_sz = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    m = gen(in_sz, hidden, out)

    # Next filename
    existing_files = list(models_dir.glob("model_small*.json"))
    numbers = [int(f.stem.replace("model_small", "")) for f in existing_files if f.stem.replace("model_small","").isdigit()]
    next_number = max(numbers, default=0) + 1
    file_path = models_dir / f"model_small{next_number}.json"

    with open(file_path, "w") as f:
        json.dump(m, f, indent=2)

    logger.info(f"Wrote {file_path}")
