

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/Model.hpp"


// ===================== STANDARD HEADERS =====================
#include <iostream>


// ===================== PROJECT HEADERS (CPU) =====================
#include "../headers/CpuLayer.hpp"
#include "../headers/cpu_loss.hpp"


template< >
void Model<cpu::Layer>::train
(
    const std::vector<std::vector<float>>& X, 
    const std::vector<std::vector<float>>& Y, 
    int epochs, float lr, int num_batches
)
{
    // Compute total batches for epoch
    const size_t total_batches = (X.size() + num_batches - 1) / num_batches;
    const size_t num_layers = layers.size();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < X.size(); i += num_batches)
        {
            // Last batch may be smaller
            size_t mini_batch = std::min(num_batches, (int)(X.size() - i));
            float batch_loss = 0.0f;

            // Accumulate gradients for each sample in the mini-batch
            for (size_t j = 0; j < mini_batch; ++j)
            {
                std::vector<std::vector<float>> A; //< Store activations
                const std::vector<float>& A_curr = X[i + j];
                const std::vector<float>& y_true = Y[i + j];

                A.push_back(A_curr); //< Save input layer as first activation

                // Forward pass
                for (const std::shared_ptr<cpu::Layer>& layer : layers)
                {
                    const std::vector<float>& curr = layer->forward(A.back(), cpu_parallel);
                    A.push_back(curr); //< Store current layer activation
                }

                // Backward pass
                std::vector<float> delta(output_size); //< delta for output layer 

                // Softmax + CrossEntropy simplification: delta = prediction - label
                for (size_t k = 0; k < delta.size(); ++k)
                {
                    delta[k] = A.back()[k] - y_true[k];
                }

                // Backpropagate through hidden layers
                for (int layer = (int)num_layers - 1; layer > 0; --layer)
                {
                    delta = layers[layer]->backward(delta, A[layer], true, cpu_parallel);
                }
                (void)layers[0]->backward(delta, A[0], false, cpu_parallel); //< input layer backward

                // Compute batch loss
                float loss = cpu::loss::cross_entropy(y_true, A.back());
                batch_loss += loss;
            }

            // Average gradients and update weights after processing the mini-batch
            for (size_t layer = 0; layer < layers.size(); ++layer)
            {
                layers[layer]->average_gradients(num_batches, cpu_parallel);
                layers[layer]->update(lr, cpu_parallel); //< SGD weight update
            }

            epoch_loss += batch_loss;

#ifdef DEBUG
            if((i / num_batches) + 1 < 10)
            std::cout << "Epoch " << epoch + 1
                << " | Batch " << (i / num_batches) + 1
                << "/" << total_batches
                << " | Avg batch loss: " << batch_loss / mini_batch
                << std::endl;
#endif

        }

        std::cout << "=== Epoch " << epoch + 1
            << " finished - Avg epoch loss: " << epoch_loss / X.size()
            << " ===\n\n";
    }

    std::cout << "CPU Training done!\n";
}

template<>
std::vector<float> Model<cpu::Layer>::inference(const std::vector<float>& X)
{
    std::vector<std::vector<float>> results;
    results.push_back(X); //< Input layer

    // Forward pass through all layers
    for (const std::shared_ptr<cpu::Layer>& layer : layers)
        results.push_back(layer->forward(results.back(), cpu_parallel));

    return results.back(); //< Return output layer
}


// ===================== CUDA HEADERS =====================
#include <cublas_v2.h>


// ===================== PROJECT HEADERS (GPU) =====================
#include "../headers/neural_layer_gpu.hpp"
#include "../headers/GpuLayer.hpp"

using Tensor = gpu::Tensor;


template<>
std::vector<float> Model<gpu::Layer>::inference(const std::vector<float>& X)
{
    cublasHandle_t handle;
    cublasCreate(&handle); //< cuBLAS context

    cudaEvent_t start, stop;
    cudaEventCreate(&start);//< GPU timing start
    cudaEventCreate(&stop); //< GPU timing stop

    // Upload input to GPU
    Tensor input(X, 1, X.size());
    input.upload();

    // Forward pass on GPU
    const Tensor* current = &input;
    for (const auto& layer : layers)
    {
        // Update current layer output
        current = &layer->forward(handle, *current, start, stop);
    }

    current->download(); //< Copy result back to host
    std::vector<float> output = current->host;

    // Cleanup GPU resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return output;
}

template< >
void Model<gpu::Layer>::train
(
    const std::vector<std::vector<float>>& X, 
    const std::vector<std::vector<float>>& Y, 
    int epochs, float lr, int num_batches
)
{
    cublasHandle_t handle;
    cublasCreate(&handle); //< cuBLAS context

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t total_batches = (X.size() + num_batches - 1) / num_batches;

    size_t in_features = X[0].size();
    size_t out_features = Y[0].size();

    // Preallocate GPU tensors for input and target
    Tensor input(num_batches, in_features);
    Tensor target(num_batches, out_features);

    for (int e = 0; e < epochs; ++e)
    {
        float epoch_loss = 0.0f;

#ifdef DEBUG
        std::cout << "\n=== [Epoch " << e + 1 << "/" << epochs << "] ===" << std::endl;
#endif
        for (size_t i = 0; i < X.size(); i += num_batches)
        {
            size_t mini_batch = std::min(num_batches, (int)(X.size() - i));

            // Upload mini-batch to GPU
            for (size_t j = 0; j < mini_batch; ++j)
            {
                input.copy_from_host_pointer(X[i + j].data(), 1, in_features, j * in_features);
                target.copy_from_host_pointer(Y[i + j].data(), 1, out_features, j * out_features);
            }

            // CPU -> GPU
            input.upload();
            target.upload();

            // Forward pass
            const Tensor* current = &input;
            for (const auto& layer : layers)
            {
                current = &layer->forward(handle, *current, start, stop);
            }

            // Compute delta for output layer
            Tensor delta(mini_batch, target.batch_size);
            gpu::compute_delta_initial
            (
                delta.dev, current->dev,           
                target.dev, mini_batch, target.batch_size       
            );

            // Backward pass
            Tensor next_delta = delta;
            for (int layer = int(layers.size()) - 1; layer > 0; layer--)
            {
                next_delta = layers[layer]->backward(handle, next_delta, start, stop);
            }
            layers[0]->backward_output(handle, next_delta, start, stop);


            // Compute batch loss
            float batch_loss = gpu::loss::cross_entropy(*current, target);
            epoch_loss += batch_loss;

            // Update weights
            for (const auto& layer : layers)
                layer->update(lr, start, stop);

#ifdef DEBUG
            if ((i / num_batches) + 1 < 10)
            std::cout << "Epoch " << e + 1
                << " | Batch " << (i / num_batches) + 1
                << "/" << total_batches
                << " | Avg batch loss: " << batch_loss / mini_batch
                << std::endl;
#endif
        }
        std::cout << "=== Epoch " << e + 1
            << " finished - Avg epoch loss: " << epoch_loss / X.size()
            << " ===" << std::endl;
    }

    // Ensure all GPU operations finished
    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    std::cout << "GPU Training done!\n";
}