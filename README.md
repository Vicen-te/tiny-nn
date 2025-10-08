
# Mini Inference (CUDA reuse) — C++20 + CUDA 12.8

Demo project: simple 2-layer FC inference engine with:
- CPU version (parallel)
- CUDA version that **reuses memory** (no cudaMalloc/cudaFree per inference)

## Requirements
- Windows 10/11
- Visual Studio 2022 (with "Desktop development with C++" workload)
- CUDA Toolkit 12.8 installed (nvcc)
- CMake >= 3.24
- Python 3 (optional, for generating models)

## Setup

1. Clone/copy this repo to your machine.
2. Download `external/json.hpp` (nlohmann json single header):
   ```powershell
   mkdir external
   curl -L  -o external/json.hpp https://github.com/nlohmann/json/releases/latest/download/json.hpp   #< --ssl-no-revoke
   ```
3. Generate a small model:
	```bash
	python data/generate_model.py 128 64 10
	```
	
## Build with Visual Studio (recommended)

- Open Visual Studio -> File -> Open -> Folder... and select the project folder.
- Visual Studio will detect CMake. Choose x64 configuration (VERY IMPORTANT).
- Build -> Build All.

Or from PowerShell / Developer Command Prompt:
	```powershell
	mkdir build
	cd build
	cmake .. -G "Visual Studio 17 2022" -A x64 -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
	cmake --build . --config Release
	```
	
## Run

From the `build/Release` folder:
	```kotlin
	mini_infer_cuda_reuse.exe ..\data\model_small.json
	```

Expected output:

- CPU vs CUDA correctness check
- Average timings per method
- results/bench.csv with timings

## Notes & Improvements

- The code uploads W and b to the GPU only once per layer. Only the initial input is uploaded per inference.
- For higher performance: reuse x_dev/y_dev without synchronizing per layer (use streams), implement batching, and use GEMM/cuBLAS.
- For profiling: use Nsight Systems / Nsight Compute.