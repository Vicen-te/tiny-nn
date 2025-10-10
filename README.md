
# Tiny-NN — Fully Connected Neural Networks in C++20 + CUDA 12.8

Tiny-NN implements fully connected neural networks with CPU and CUDA support. It provides:
- CPU execution (parallelized)
- CUDA execution with memory reuse (weights and biases uploaded only once per layer)
- Training with backpropagation and SGD
- Model serialization using [`json.hpp`](https://github.com/nlohmann/json) (MIT licensed) included in the repository

## Requirements
- C++20 compiler
- CUDA 12.8 installed (for GPU support)
- CMake >= 3.24
- Python 3.12 (optional, for dataset download and preview)

> Although development was done on Windows 10/11 using Visual Studio 2022, the project can be built on any OS with a compatible C++20 compiler and CUDA installation.

## Setup

1. Clone or copy the repository to your machine.
```bash
git clone https://github.com/Vicen-te/tiny-nn.git
cd tiny-nn
```

2. Download the MNIST dataset:
Using Python script (recommended):
```bash
python scripts/download_mnist.py
```
- This will download and save the MNIST dataset in `data/minst/`.
- Alternatively, you can download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

3. Optional: generate a small model using Python (arguments: `input layer`, `hidden layer`, `output layer`):
```bash
python data/generate_model.py 128 64 10
```
	
4. Optional: preview MNIST digits:

- Python: `python scripts/preview.py`
- C++: `ascii_preview()` function in MNISTLoader

	
## Build with Visual Studio:

- Open Visual Studio -> File -> Open -> Folder... and select the project folder.
- Visual Studio will detect CMake. For GPU usage, choose x64 configuration.
- Build -> Build All.

### Or from PowerShell / Developer Command Prompt (recommended):
#### Option 1: Specify all options manually
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
cmake --build . --config Release
```

- `-G "Visual Studio 17 2022"` selects Visual Studio 2022
- `-A x64` selects 64-bit architecture (recommended for CUDA)
- `-DCUDA_TOOLKIT_ROOT_DIR` is optional, CMake can auto-detect CUDA
> Note: The -A x64 option is recommended if you want to use CUDA on Windows. On Linux or macOS, this is not necessary.

#### Option 2: Let CMake detect everything automatically (recommended)
```powershell
cmake -B build -S .
cmake --build build --config Release
```

- CMake will detect Visual Studio and CUDA if installed in standard locations
- `-S` is the source folder, `-B` is the build folder
>Both methods produce the same result. Use Option 2 for simplicity and fewer manual settings.

	
## Run

From the `build/Release` folder:
```kotlin
tinny-nn.exe
```

Expected output:

- CPU vs CUDA correctness check
- Average timings per method
- `results/bench.csv` with timings
- Training progress (if enabled)
- ASCII preview of MNIST samples (if enabled)

## Notes & Improvements

- Currently, weights `W` and biases `b` are uploaded to the GPU **once per layer**. The input vector is uploaded for each inference.  
- Intermediate GPU buffers (`dX`/`dY`) are allocated per layer, but **are not fully reused** across inferences or layers.  
- For higher performance (future improvements):  
  - Reuse intermediate GPU buffers without synchronizing per layer (use CUDA streams).  
  - Implement batching to process multiple inputs simultaneously.  
  - Consider replacing the simple FC kernel with GEMM/cuBLAS for faster matrix multiplication.  
- Profiling can be done with Nsight Systems / Nsight Compute.