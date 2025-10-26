

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/GpuLayer.hpp"


// ===================== STANDARD HEADERS =====================
#include <cublas_v2.h>


// ===================== PROJECT HEADERS =====================
#include "../headers/neural_layer_gpu.hpp"


namespace gpu
{
 
    void Layer::init()
    {
        // Upload weights to GPU
        T_W.copy_from_host_pointer(W.ptr(), out, in);
        T_W.upload();

        // Upload biases to GPU
        T_b.copy_from_host_pointer(b.data(), 1, out);
        T_b.upload();

        // Create CUDA stream for asynchronous GPU operations
        cudaStreamCreate(&stream);
    }

    void Layer::update_W_b()
    {
        W.copy_from_vector(T_W.host);
        std::memcpy(b.data(), T_b.host.data(), T_b.host.size() * sizeof(float));
    }

    Layer::Layer
    (
        size_t input_dim,
        size_t output_dim,
        ActivationType activation_type,
        std::string_view layer_name
    )
        : LayerBase(input_dim, output_dim, activation_type, layer_name)
    {
        init();
    }

    Layer::~Layer()
    {
        cudaStreamDestroy(stream);
    }

    void Layer::from_json(const json& layer)
    {
        LayerBase::from_json(layer);
        init();
    }

    const Tensor& Layer::forward
    (
        cublasHandle_t handle,
        const Tensor& input,
        cudaEvent_t start,
        cudaEvent_t stop
    )
    {
        // Resize GPU buffers if batch size changed
        X.ensure_size(input.batches, in);
        X.copy_from_device_only(input);
        Y.ensure_size(input.batches, out);
        X.synchronize();
        
        // Optionally record timing start
        if (start) cudaEventRecord(start, stream);

        // Compute: Y = activation(X * W + b)
        layer::forward
        (
            handle, 
            stream,
            T_W.dev, 
            T_b.dev, 
            X.dev, 
            Y.dev,
            in, 
            out, 
            input.batches, 
            activation
        );

        // Optionally record timing end
        if (stop) cudaEventRecord(stop, stream);

        return Y;
    }

    const Tensor Layer::backward
    (
        cublasHandle_t handle,
        Tensor delta,
        cudaEvent_t start,
        cudaEvent_t stop
    )
    {
        // Allocate gradient tensors
        Tensor dX(delta.batches, in);
        dW.ensure_size(out, in);
        db.ensure_size(1, out);

        // Optional timing start
        if (start) cudaEventRecord(start, stream);

        // Apply activation derivative to delta
        layer::activation_backward
        (
            stream,
            delta.dev,      
            Y.dev,          
            delta.batches,  
            out,            
            activation      
        );
        
        // Compute gradients for weights, biases, and previous activations
        layer::backward
        (
            handle, stream,
            T_W.dev,        
            X.dev,          
            delta.dev,      
            dX.dev,         
            dW.dev,         
            db.dev,         
            in, out, delta.batches
        );

        // Optional timing stop
        if (stop) cudaEventRecord(stop, stream);

        return dX;
    }

    void Layer::backward_output
    (
        cublasHandle_t handle,
        Tensor delta, // Para Softmax+CrossEntropy, esto será y_true
        cudaEvent_t start,
        cudaEvent_t stop
    )
    {
        dW.ensure_size(out, in);
        db.ensure_size(1, out);

        // Optional timing start
        if (start) cudaEventRecord(start, stream);

        // Apply activation derivative
        layer::activation_backward
        (
            stream,
            delta.dev,         
            Y.dev,             
            delta.batches,     
            out,               
            activation         
        );

        // Compute parameter gradients specific to the output layer
        layer::backward_output
        (
            handle,
            stream,
            X.dev,        
            delta.dev,    
            dW.dev,       
            db.dev,       
            in, out, delta.batches
        );

        // Optional timing stop
        if (stop) cudaEventRecord(stop, stream);
    }

    void Layer::update
    (
        const float& lr,
        cudaEvent_t start,
        cudaEvent_t stop
    )
    {
        // Optional timing start
        if (start) cudaEventRecord(start, stream);

        // Update GPU weights and biases using SGD
        layer::sgd_update(stream, T_W.dev, dW.dev, lr, in * out);
        layer::sgd_update(stream,  T_b.dev, db.dev, lr, out);

        // Optional timing stop
        if (stop) cudaEventRecord(stop, stream);

        // Synchronize updated parameters with CPU
        T_W.download();
        T_b.download();
        update_W_b();

        // Release temporary gradient memory
        dW.free_memory();
        db.free_memory();
    }

}