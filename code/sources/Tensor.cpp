

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/Tensor.hpp"


// ===================== STANDARD HEADERS =====================
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <format>


namespace
{
    /**
     * @brief Checks the status of a CUDA call and throws an exception if it fails.
     *
     * @param err The error code returned by a CUDA function.
     * @param file The source file where the call is made (default: __FILE__).
     * @param line The line number in the source file (default: __LINE__).
     *
     * @throws std::runtime_error if the CUDA call did not succeed.
     */
    inline void check_error
    (
        cudaError_t err,
        std::string_view file = __FILE__,
        int line = __LINE__
    )
    {
        if (err != cudaSuccess) [[unlikely]] 
        {
            throw std::runtime_error
            (
                std::format("CUDA error at {}:{} - {}", file, line, cudaGetErrorString(err))
            );
        }
    }
}

namespace gpu
{

    void CudaDeleter::operator()(float* ptr) const
    {
        // Free GPU memory when unique_ptr goes out of scope
        if (ptr) check_error(cudaFree(ptr));
    }

    void Tensor::copy_from_tensor(const Tensor& src)
    {
        if (!src.is_allocated()) [[unlikely]]
            throw std::runtime_error("copy_from_tensor: source tensor not allocated");

        // Copy CPU data first (fast memcpy on host)
        copy_from_host_only(src);

        // Then copy GPU data (device-to-device)
        copy_from_device_only(src);
    }

    Tensor::Tensor
    (
        size_t num_batches, 
        size_t elements_per_batch
    ) 
    {
        // Allocate both host and device memory
        allocate(num_batches, elements_per_batch);
    }

    Tensor::Tensor
    (
        const std::vector<float>& data, 
        size_t num_batches, 
        size_t elements_per_batch
    ) 
    {
        if (data.size() != num_batches * elements_per_batch) [[unlikely]]
            throw std::runtime_error("Tensor: host data size mismatch");

        // Reserve required GPU and CPU memory
        allocate(num_batches, elements_per_batch);

        // Copy vector contents directly into the vector's internal contiguous buffer
        std::memcpy(host.data(), data.data(), data.size() * sizeof(float));
    }

    Tensor::Tensor(const Tensor& source)
    {
        // Allocate memory equal to source tensor
        allocate(source.batches, source.batch_size);

        // Copy data from both host and device memory
        copy_from_tensor(source);
    }

    Tensor& Tensor::operator=(const Tensor& source)
    {
        if (this == &source) return *this; //< Prevent self-assignment

        // Release current memory before reallocating
        free_memory();

        // Allocate same memory layout as source
        allocate(source.batches, source.batch_size);

        // Copy both CPU and GPU buffers
        copy_from_tensor(source);

        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept
    {
        if (this != &other)
        {
            free_memory(); //< Clean current memory
            move_from(std::move(other)); //< Transfer ownership (no deep copy)
        }
        return *this;
    }

    void Tensor::allocate(size_t num_batches, size_t elements_per_batch)
    {
        if (dev) free_memory(); //< Prevent memory leaks

        batches = num_batches;
        batch_size = elements_per_batch;

        // Resize host buffer to match new shape
        host.resize(num_batches * elements_per_batch);

        // Allocate device (GPU) memory
        check_error(cudaMalloc
        (
            reinterpret_cast<void**>(&dev), 
            num_batches * elements_per_batch * sizeof(float)
        ));

        // Stream creation could be added here if needed
        // TODO: if (!stream) check_error(cudaStreamCreate(&stream));
    }

    void Tensor::ensure_size(size_t num_batches, size_t elements_per_batch)
    {
        // Reallocate only if the total number of elements has changed
        if (num_batches * elements_per_batch != batches * batch_size) 
            allocate(num_batches, elements_per_batch);
    }

    void Tensor::resize(size_t num_batches, size_t elements_per_batch)
    {
        // If dimensions change, reallocate; otherwise just update members
        if (num_batches * elements_per_batch != batches * batch_size) 
            allocate(num_batches, elements_per_batch);
        else
        {
            batches = num_batches; 
            batch_size = elements_per_batch; 
        }
    }

    void Tensor::copy_from_host_pointer
    (
        const float* ptr, 
        size_t num_batches, 
        size_t elements_per_batch, 
        size_t offset
    )
    {
        if (!ptr) [[unlikely]] 
            throw std::runtime_error("from_host_matrix: host_ptr is null");

        const size_t total = num_batches * elements_per_batch;

        // Ensure buffer is large enough to hold new data at given offset
        if (offset + total > host.size())
            ensure_size(offset / elements_per_batch + num_batches, elements_per_batch);

        // Copy data from external pointer into vector's internal storage at the given offset
        std::memcpy(host.data() + offset, ptr, total * sizeof(float));
    }

    void Tensor::copy_from_device_only(const Tensor& source)
    {
        if (!source.is_allocated()) [[unlikely]]
            throw std::runtime_error("copy_from: source tensor not allocated");

        // Perform asynchronous GPU-to-GPU copy
        check_error
        (
            cudaMemcpyAsync
            (
                dev.get(),
                source.dev.get(),
                source.batches * source.batch_size * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            )
        );

        // Check for errors in async copy
        check_error(cudaGetLastError()); 
    }

    void Tensor::copy_from_host_only(const Tensor& source)
    {
        if (source.host.empty()) [[unlikely]]
            throw std::runtime_error("copy_from_host_only: source host data is empty");

        // Match source host size before copying
        host.resize(source.host.size());

        // Perform fast CPU-to-CPU copy
        std::memcpy(host.data(), source.host.data(), host.size() * sizeof(float));
    }

    void Tensor::upload() const
    {
        if (!dev || host.empty()) return;

        // Asynchronous transfer: host -> device
        check_error
        (
            cudaMemcpyAsync
            (
                dev.get(),
                host.data(),
                batches * batch_size * sizeof(float),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // Check for errors in async copy
        check_error(cudaGetLastError());
    }

    void Tensor::download() const
    {
        if (!dev || host.empty()) return;

        // Asynchronous transfer: device -> host
        check_error
        (
            cudaMemcpyAsync
            (
                const_cast<float*>(host.data()),
                dev.get(),
                batches * batch_size * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            )
        );

        // Check for errors in async copy
        check_error(cudaGetLastError());
    }

    void Tensor::synchronize() const
    {
        if (stream) [[unlikely]]
        {
            // Wait for all queued operations in the stream to finish
            check_error(cudaStreamSynchronize(stream));
        }
        else [[likely]]
        {
            // Block until all device operations complete
            check_error(cudaDeviceSynchronize());
        }
    }

    void Tensor::clearDevice()
    {
        if (dev)
        {
            // Fill GPU memory with zeros asynchronously
            check_error
            (
                cudaMemsetAsync
                (
                    dev.get(),
                    0,
                    batches * batch_size * sizeof(float),
                    stream
                )
            );

            // Check for errors in async memset
            check_error(cudaGetLastError());
        }
    }

    void Tensor::free_memory()
    {
        // Release GPU memory via unique_ptr deleter
        dev.reset();

        if (stream) [[unlikely]]
        {
            // Ensure stream has finished all operations before destroying
            check_error(cudaStreamSynchronize(stream));
            check_error(cudaStreamDestroy(stream));
            stream = nullptr;
        }
        else [[likely]]
            check_error(cudaDeviceSynchronize());

        // Check for any pending CUDA errors
        check_error(cudaGetLastError());

        // Release CPU memory and reset members
        host.clear();
        batches = batch_size = 0;
    }

    Tensor::~Tensor()
    {
        // Automatic cleanup on destruction
        free_memory();
    }
}