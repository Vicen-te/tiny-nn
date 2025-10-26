

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/neural_layer_gpu.hpp"


// ===================== CUDA HEADERS =====================
#include <cuda_runtime.h>
#include <cublas_v2.h>


// ===================== STANDARD HEADERS =====================
#include <iostream>
#include <format>
#include <string_view>


// ===================== PROJECT HEADERS =====================
#include "../headers/kernels_cuda.cuh"


namespace
{

    const constinit int min_threads_per_block = 256;

    /**
     * @brief Calculate optimal CUDA grid and block dimensions for a 1D kernel.
     *
     * @param total Total number of threads required.
     * @param min_threads_per_block Minimum threads per block (default 128).
     * @return std::pair<dim3, dim3> {block, grid}
     */
    inline const std::pair<dim3, dim3> compute_grid_block(int total) 
    {
        // Query maximum threads per block for device 0
        int max_threads_per_block = 0;
        cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

        // Choose threads per block (clamped between min and max)
        int threads = std::min(std::max(min_threads_per_block, 1), max_threads_per_block);

        // Compute number of blocks needed (ceil division)
        int blocks = (total + threads - 1) / threads;

        return { dim3(threads), dim3(blocks) };
    }

    /**
     * @brief Check cuBLAS call status and report any errors.
     *
     * @param status The cuBLAS status returned by a function.
     * @param context Optional string describing where the error occurred.
     */
    inline void check_cuBLAS_error(cublasStatus_t status, std::string_view context = "cuBLAS SGEMM") noexcept
    {
        if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]]
        {
            std::cerr << std::format("{} failed with status code: {}\n", context, static_cast<int>(status));
        }
    }

    /**
     * @brief Check and report the last CUDA kernel launch error.
     *
     * Should be called after a kernel execution to detect runtime launch failures.
     */
    inline void check_cuda_kernel_error(std::string_view context = "CUDA kernel launch") noexcept
    {
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) [[unlikely]]
        {
            std::cerr << std::format("{} error: {}\n", context, cudaGetErrorString(err));
        }
    }

}


namespace gpu::layer
{

    void forward
    (
        cublasHandle_t handle,
        cudaStream_t stream,
        const std::unique_ptr<float, CudaDeleter>& W,         
        const std::unique_ptr<float, CudaDeleter>& b,         
        const std::unique_ptr<float, CudaDeleter>& X,         
        std::unique_ptr<float, CudaDeleter>& Y,               
        const size_t& in,
        const size_t& out,
        const size_t& batch,
        ActivationType act
    )
    {
        // Convert sizes to int for cuBLAS calls
        const int in_i = static_cast<int>(in);
        const int out_i = static_cast<int>(out);
        const int batch_i = static_cast<int>(batch);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Compute total elements and optimal grid/block
        const int total = batch_i * out_i;
        auto [block, grid] = compute_grid_block(total);

        // Set cuBLAS stream for SGEMM
        cublasSetStream(handle, stream);

        // row-major 
        // Y = X * W^T

        // column-major
        // Y^T = W * X^T

        // X [batch x in], W [out x in], Y [batch x out]
        // T: X [in x batch], W [in x out], Y [out x batch]

        // C = A * B 
        // Y^T = W^T^T * X^T
        cublasStatus_t status = cublasSgemm
        (
            handle,
            CUBLAS_OP_T,    // A^T
            CUBLAS_OP_N,    // B 
            out_i,          // m = rows of C
            batch_i,        // n = cols of C
            in_i,           // k = rows of B != T o cols of A != T
            &alpha,
            W.get(), in_i,  // lda = rows of A
            X.get(), in_i,  // ldb = rows of B
            &beta,
            Y.get(), out_i  // ldc = rows of C
        );

        check_cuBLAS_error(status, "Forward SGEMM");

        // Launch kernel to add bias to each output
        kernels::add_bias << <grid, block, 0, stream >> >
            (
                Y.get(),
                b.get(),
                batch_i,
                out_i
                );

        // Apply activation function: softmax or element-wise
        if (act == ActivationType::SOFTMAX)
        {
            dim3 blocks(batch_i); //< one block per row
            dim3 threads(1);      //< one thread per block  
            kernels::softmax_per_row << <blocks, threads, 0, stream >> >
                (
                    Y.get(),
                    batch_i,
                    out_i
                    );
        }
        else
        {
            kernels::apply_activation << < grid, block, 0, stream >> >
                (
                    Y.get(),
                    total,
                    act
                    );
        }

        check_cuda_kernel_error("Forward kernel launch");
    }

    void backward
    (
        cublasHandle_t handle,
        cudaStream_t stream,
        std::unique_ptr<float, CudaDeleter>& W,
        const std::unique_ptr<float, CudaDeleter>& A_prev,  
        std::unique_ptr<float, CudaDeleter>& delta,         
        std::unique_ptr<float, CudaDeleter>& dX,            
        std::unique_ptr<float, CudaDeleter>& dW,            
        std::unique_ptr<float, CudaDeleter>& dB,            
        const size_t& in,                                   
        const size_t& out,                                  
        const size_t& batch                                 
    )
    {
        // Convert sizes to int for cuBLAS calls
        const int in_i = static_cast<int>(in);
        const int out_i = static_cast<int>(out);
        const int batch_i = static_cast<int>(batch);

        const float alpha_dw = 1.0f / batch; //< scale weight gradient by batch
        const float alpha_dx = 1.0f;
        const float beta = 0.0f;

        const int total = batch_i * out_i;
        auto [block, grid] = compute_grid_block(total);

        cublasSetStream(handle, stream);

        // dB = sum(delta) across the batch
        kernels::accumulate_bias_gradient << < grid, block, 0, stream >> >
            (
                delta.get(),
                dB.get(),
                batch_i,
                out_i
                );

        check_cuda_kernel_error("Bias gradient kernel");

        // row-major 
        // dW = delta^T * A_prev 

        // column-major
        // dW^T = A_prev^T * delta 

        // delta [batch x out], A_prev [batch x in], dW [out x in]
        // T: delta [out x batch], A_prev [in x batch], dW [in x out]

        // C = A * B 
        // dW^T = A_prev^T * delta^T^T
        cublasStatus_t status = cublasSgemm
        (
            handle,
            CUBLAS_OP_N,        // A
            CUBLAS_OP_T,        // B^T 
            in_i,               // m = rows of C
            out_i,              // n = cols of C
            batch_i,            // k = rows of B != T o cols of A != T
            &alpha_dw,
            A_prev.get(), in_i, // lda = rows of A
            delta.get(), out_i, // ldb = rows of B
            &beta,
            dW.get(), in_i      // ldc = rows of C
        );

        check_cuBLAS_error(status, "Weight gradient SGEMM");

        // row-major 
        // delta_prev = delta * W * d_activation (applied later)

        // column-major
        // delta_prev^T = W^T * delta^T

        // W [out x in], delta [batch x out], delta_prev [batch x in], d_activation [batch x in]                    
        // T: W [in x out], delta [out x batch], delta_prev [in x batch]

        // C = A * B 
        // delta_prev^T = W^T * delta^T
        status = cublasSgemm
        (
            handle,
            CUBLAS_OP_N,        // A
            CUBLAS_OP_N,        // B
            in_i,               // m = rows of C
            batch_i,            // n = cols of C
            out_i,              // k = rows of B != T o cols of A != T
            &alpha_dx,
            W.get(), in_i,      // lda = rows of A
            delta.get(), out_i, // ldb = rows of B
            &beta,
            dX.get(), in_i      // ldc = rows of C
        );

        check_cuBLAS_error(status, "Backward propagation SGEMM");
    }

    void backward_output
    (
        cublasHandle_t handle,
        cudaStream_t stream,
        const std::unique_ptr<float, CudaDeleter>& A_prev,   
        std::unique_ptr<float, CudaDeleter>& delta,          
        std::unique_ptr<float, CudaDeleter>& dW,             
        std::unique_ptr<float, CudaDeleter>& dB,             
        const size_t& in,                                    
        const size_t& out,                                   
        const size_t& batch                                  
    )
    {
        const int in_i = static_cast<int>(in);
        const int out_i = static_cast<int>(out);
        const int batch_i = static_cast<int>(batch);

        const float alpha = 1.0f / batch;
        const float beta = 0.0f;

        const int total = batch_i * out_i;
        auto [block, grid] = compute_grid_block(total);

        cublasSetStream(handle, stream);

        // dB = sum(delta) for the input layer 
        // input layer: (first layer in forward, last processed in backward)
        kernels::accumulate_bias_gradient << < grid, block, 0, stream >> >
            (
                delta.get(),
                dB.get(),
                batch_i,
                out_i
                );

        check_cuda_kernel_error("Bias gradient kernel (output)");

        // row-major 
        // dW = delta^T * A_prev 

        // column-major
        // dW^T = A_prev^T * delta 

        // delta [batch x out], A_prev [batch x in], dW [out x in]
        // T: delta [out x batch], A_prev [in x batch], dW [in x out]

        // C = A * B 
        // dW^T = A_prev^T * delta^T^T
        cublasStatus_t status = cublasSgemm
        (
            handle,
            CUBLAS_OP_N,        // A
            CUBLAS_OP_T,        // B^T 
            in_i,               // m = rows of C
            out_i,              // n = cols of C
            batch_i,            // k = rows of B != T o cols of A != T
            &alpha,
            A_prev.get(), in_i, // lda = rows of A
            delta.get(), out_i, // ldb = rows of B
            &beta,
            dW.get(), in_i      // ldc = rows of C
        );


        check_cuBLAS_error(status, "Output layer SGEMM");
    }

    void activation_backward
    (
        cudaStream_t stream,
        std::unique_ptr<float, CudaDeleter>& delta,   
        const std::unique_ptr<float, CudaDeleter>& Y, 
        const size_t& batch,                          
        const size_t& out,                            
        ActivationType act                            
    )
    {
        const int total = static_cast<int>(batch * out);
        auto [block, grid] = compute_grid_block(total);

        // Apply derivative of activation function element-wise to delta
        // (W^T * delta^T) * activation_derivative(Y)
        kernels::apply_activation_derivative << < grid, block, 0, stream >> >
        (
            delta.get(),
            Y.get(),
            total,
            act
        );

        check_cuda_kernel_error("Activation derivative kernel");
    }

    void sgd_update
    (
        cudaStream_t stream,
        std::unique_ptr<float, CudaDeleter>& param,
        std::unique_ptr<float, CudaDeleter>& grad,
        const float& lr,
        const size_t& size
    )
    {
        const int size_i = static_cast<int>(size);
        auto [block, grid] = compute_grid_block(size_i);

        // Update parameters using SGD: param -= lr * grad
        kernels::update_parameters << < grid, block, 0, stream >> >
            (
                param.get(),
                grad.get(),
                lr,
                size_i
                );

        check_cuda_kernel_error("SGD update kernel");
    }

}


namespace gpu::loss
{

    float cross_entropy
    (
        const Tensor& y_pred,      
        const Tensor& y_true       
    )
    {
        const int batch = static_cast<int>(y_pred.batches);
        const int out = static_cast<int>(y_pred.batch_size);

        const int total = batch * out;
        auto [block, grid] = compute_grid_block(total);

        // Preallocate temporary tensor for block-wise loss reduction
        static Tensor d_block_losses;
        if (!d_block_losses.is_allocated() || d_block_losses.batches != grid.x)
            d_block_losses.allocate(grid.x, 1);

        // Compute cross-entropy per block
        kernels::cross_entropy_per_block <<< grid, block, block.x * sizeof(float) >>>
        (
            y_pred.dev.get(), 
            y_true.dev.get(), 
            d_block_losses.dev.get(),
            total
        );

        check_cuda_kernel_error("Cross-entropy kernel");

        // Download per-block losses to host and sum them
        d_block_losses.download();
        float total_loss = 0.0f;
        for (int i = 0; i < static_cast<int>(grid.x); ++i)
            total_loss += d_block_losses.host[i];

        return total_loss;
    }

}


namespace gpu
{

    void compute_delta_initial
    (
        std::unique_ptr<float, CudaDeleter>& delta,        
        const std::unique_ptr<float, CudaDeleter>& Y,      
        const std::unique_ptr<float, CudaDeleter>& y_true, 
        const size_t& batch,
        const size_t& out
    )
    {
        const int total = static_cast<int>(batch * out);
        auto [block, grid] = compute_grid_block(total);

        // Compute initial delta for 
        // softmax + cross-entropy: delta = y_pred - y_true
        kernels::delta_softmax_crossentropy <<< grid, block, 0 >>>
        (
            Y.get(),
            y_true.get(),
            delta.get(),
            total
        );

        check_cuda_kernel_error("Delta initial kernel");
    }

}