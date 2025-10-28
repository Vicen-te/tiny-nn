# ===================== Build =====================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

# Install dependencies (Ubuntu 24.04 includes GCC 13 and CMake 3.28+ with full C++20 support)
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git zip clang-tidy cppcheck \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

# Build the project
RUN mkdir -p build/Release && cd build/Release && \
    rm -f CMakeCache.txt && \
    cmake ../../ -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
        -DCMAKE_CUDA_ARCHITECTURES="all" && \
    ninja && \
    # Run benchmark and save output, ignore errors if benchmark fails
    mkdir -p release && \
    ./tiny_nn > release/tiny_nn_benchmark.json || true && \
    # Package the build output
    zip -r tiny_nn_release.zip release

# ===================== Runtime =====================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

WORKDIR /app

# Copy the compiled executable and benchmark results from the build stage
COPY --from=build /workspace/build/Release/tiny_nn ./build/tiny_nn
COPY --from=build /workspace/build/Release/release ./build/release

WORKDIR /app/build

# Set the default entrypoint
ENTRYPOINT ["./tiny_nn"]
