import json, random, sys, re
from pathlib import Path
import logging


def gen(in_sz=128, hidden=64, out=10, seed=123):
    """Generate a simple 2-layer fully connected neural network model with random weights and biases."""
    random.seed(seed)
    layers = []

    # First fully connected layer
    w1 = [random.uniform(-0.5,0.5) for _ in range(hidden*in_sz)]
    b1 = [random.uniform(-0.1,0.1) for _ in range(hidden)]
    layers.append({
        "type":"fc","name":"fc1","in":in_sz,"out":hidden,
        "weights": w1, "bias": b1, "activation":"relu"
    })

    # Second fully connected layer
    w2 = [random.uniform(-0.5,0.5) for _ in range(out*hidden)]
    b2 = [random.uniform(-0.1,0.1) for _ in range(out)]
    layers.append({
        "type":"fc","name":"fc2","in":hidden,"out":out,
        "weights": w2, "bias": b2, "activation":"linear"
    })

    return {"layers": layers}


if __name__ == "__main__":
    # Directory to save model files
    root = Path(__file__).parent.parent
    models_dir = root / "data" / "models"
    models_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )
    logger = logging.getLogger()

    # Get model parameters from command line arguments (with defaults)
    in_sz = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Generate the model
    m = gen(in_sz, hidden, out)

    # Find all existing model_small*.json files
    existing_files = list(models_dir.glob("model_small*.json"))

    numbers = []
    pattern = re.compile(r"model_small(\d+)$")  # Match 'model_small' followed by digits at the end

    for f in existing_files:
        match = pattern.search(f.stem)
        if match:
            numbers.append(int(match.group(1)))

    # Determine the next number
    next_number = max(numbers, default=0) + 1

    # Create the filename
    file_path = models_dir / f"model_small{next_number}.json"

    # Save the model to a JSON file
    with open(file_path, "w") as f:
        json.dump(m, f, indent=2)

    # Log the output path
    logger.info(f"Wrote {file_path}")
