

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <vector>
#include <memory>
#include <stdexcept>


// Forward declarations for CUDA/cuBLAS types
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;


namespace gpu
{

    // ===================== CUSTOM GPU MEMORY DEALLOCATOR =====================
    /**
     * @brief Functor used as a custom deleter for GPU memory.
     * 
     * Automatically calls cudaFree when the pointer goes out of scope.
     */
    struct CudaDeleter 
    {
        void operator()(float* ptr) const;
    };


     // ===================== TENSOR CLASS =====================
    /**
     * @brief Lightweight CPU-GPU tensor wrapper for float data.
     *
     * Manages memory both on host (CPU) and device (GPU), providing convenient
     * methods for allocation, upload/download between host and device, and
     * synchronization. Designed for batch operations in neural networks.
     */
    class Tensor
    {

    private:


        // ===================== INTERNAL HELPERS =====================
        /**
         * @brief Copy both host and device data from another tensor.
         * @param src Source tensor
         */
        void copy_from_tensor(const Tensor& src);

        /**
         * @brief Helper for move constructor and move assignment.
         * 
         * Transfers ownership of GPU memory and host buffer.
         */
        inline void move_from(Tensor&& other) noexcept
        {
            host = std::move(other.host);
            dev = std::move(other.dev);
            batches = other.batches;
            batch_size = other.batch_size;
            stream = other.stream;

            other.dev = nullptr;
            other.batches = other.batch_size = 0;
        }



    public:


        // ===================== MEMBERS =====================
        /// GPU buffer with custom deleter
        std::unique_ptr<float, CudaDeleter> dev = nullptr;
        std::vector<float> host;        ///< CPU buffer 
        size_t batches = 0;             ///< Number of batches
        size_t batch_size = 0;          ///< Elements of elements per batch
        cudaStream_t stream = nullptr;  ///< Optional CUDA stream for async operations


        // ===================== CONSTRUCTORS & ASSIGNMENT OPERATORS =====================

        Tensor() = default; ///< Default constructor (does not allocate memory)

        /**
         * @brief Allocate tensor memory on host and device.
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         */
        Tensor(size_t num_batches, size_t elements_per_batch);

        /**
         * @brief Create tensor from host vector.
         * @param data Host data to copy
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         */
        Tensor
        (
            const std::vector<float>& data, 
            size_t num_batches, 
            size_t elements_per_batch
        );


        Tensor(const Tensor& source);               ///< Copy constructor
        Tensor& operator=(const Tensor&);           ///< Copy assignment
        Tensor& operator=(Tensor&& other) noexcept; ///< Move assignment
        Tensor(Tensor&& other) noexcept             ///< Move constructor
        {
            move_from(std::move(other));
        }


        // ===================== MEMORY MANAGEMENT =====================
        /**
         * @brief Allocate host and device memory for the tensor. 
         * 
         * Frees previous memory if allocated.
         * 
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         */
        void allocate(size_t num_batches, size_t elements_per_batch);

        /**
         * @brief Ensure tensor has at least the given size, reallocates if needed.
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         */
        void ensure_size(size_t num_batches, size_t elements_per_batch);

        /**
         * @brief Resize tensor while preserving device memory if possible.
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         */
        void resize(size_t num_batches, size_t elements_per_batch);


        // ===================== DATA TRANSFER =====================
        /**
         * @brief Copy data from a host pointer into the tensor.
         * @param ptr Pointer to host data
         * @param num_batches Number of mini-batches
         * @param elements_per_batch Number of elements in each batch
         * @param offset Offset in the tensor host buffer
         */
        void copy_from_host_pointer(const float* ptr, size_t num_batches, size_t elements_per_batch, size_t offset = 0);

        /**
         * @brief Copy only device data from another tensor asynchronously.
         * @param source Tensor from which to copy device memory
         */
        void copy_from_device_only(const Tensor& source);


        /**
         * @brief Copy only host data from another tensor.
         * @param source Tensor from which to copy host memory
         */
        void copy_from_host_only(const Tensor& source);


        /**
         * @brief Upload host buffer to GPU asynchronously if stream is set.
         */
        void upload() const;

        /**
         * @brief Download GPU buffer to host asynchronously if stream is set.
         */
        void download() const;


        // ===================== SYNCHRONIZATION =====================
        /**
         * @brief Synchronize the CUDA stream or the device if stream is null.
         */
        void synchronize() const;


        // ===================== UTILITIES =====================
         /**
         * @brief Clear GPU memory without touching host data.
         */
        void clearDevice();

        /**
         * @brief Fill host buffer with a constant value.
         * @param value Value to fill
         */
        void fill(float value)
        {
            std::fill(host.begin(), host.end(), value);
        }


        // ===================== ACCESSORS =====================
        /**
         * @brief Check if GPU memory is allocated.
         * @return True if GPU memory exists
         */
        [[nodiscard]] bool is_allocated() const noexcept
        { 
            return dev != nullptr; 
        }


        // ===================== RESOURCE MANAGEMENT =====================
        /**
         * @brief Free GPU memory and clear host buffer.
         */
        void free_memory();

        ~Tensor(); ///< Destructor releases GPU memory automatically

    };

}