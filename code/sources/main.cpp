
// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <string>

#include "json.hpp"
#include "../headers/model.hpp"
#include "../headers/utils.hpp"
#include "../headers/tensor.hpp"
#include "../headers/fc_cpu.hpp"
#include "../headers/fc_cuda.hpp"

using json = nlohmann::json;

// Undefine DEBUG to disable debug prints in this file
#ifdef DEBUG
#undef DEBUG
#endif

/**
 * @brief Load a fully connected model from a JSON file.
 * @param path Path to the JSON model file
 * @return Model object
 */
static Model load_model(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Failed to open model file: " + path);

    json json_model;
    file >> json_model;

    Model model;
    for (json& json_layer : json_model["layers"])
    {
        Layer layer;
        layer.name = json_layer.value("name", "");
        layer.in = json_layer["in"].get<size_t>();
        layer.out = json_layer["out"].get<size_t>();
        layer.activation = json_layer.value("activation", "linear");

        // Allocate memory for weights and biases
        layer.W.resize(layer.out, layer.in);
        layer.b.assign(layer.out, 0.0f);

        // Load weights from JSON
        json json_weights = json_layer["weights"];
        if (json_weights.size() != layer.out * layer.in)
            throw std::runtime_error("Weight size mismatch in layer: " + layer.name);

        for (size_t i = 0; i < layer.out; ++i)
            for (size_t k = 0; k < layer.in; ++k)
                layer.W(i, k) = json_weights[i * layer.in + k].get<float>();

        // Load bias from JSON
        json json_bias = json_layer["bias"];
        if (json_bias.size() != layer.out)
            throw std::runtime_error("Bias size mismatch in layer: " + layer.name);

        for (size_t i = 0; i < layer.out; ++i)
            layer.b[i] = json_bias[i].get<float>();

        model.layers.push_back(std::move(layer));
    }

    // Set input and output sizes based on first and last layer
    if (!model.layers.empty())
    {
        model.input_size = model.layers.front().in;
        model.output_size = model.layers.back().out;
    }

    return model;
}

int main(int argc, char** argv)
{
    // --- Parse model path and benchmark CSV path from command line ---
    std::string base = PROJECT_SOURCE_DIR;
    std::string model_path = base + "/data/models/model_small1.json";
    std::string bench_path = base + "/results/bench.csv";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) bench_path = argv[2];

    std::cout << "Loading model from: " << model_path << "\n";
    Model model = load_model(model_path);
    std::cout << "Model info: input="
        << model.input_size
        << " | layers=" << model.layers.size()
        << " | output=" << model.output_size << "\n";

    for (size_t i = 0; i < model.layers.size(); ++i)
        std::cout << "Layer " << i << ": in=" << model.layers[i].in
        << ", out=" << model.layers[i].out << "\n";

    // --- Generate random input vector ---
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(model.input_size);
    for (auto& val : input) val = dist(rng);

    // === CUDA Setup ===
    // Prepare GPU tensors for weights, biases, and intermediate buffers
    std::vector<Tensor> dW(model.layers.size());
    std::vector<Tensor> dB(model.layers.size());
    std::vector<Tensor> dX(model.layers.size());
    std::vector<Tensor> dY(model.layers.size());

    for (size_t layer_idx = 0; layer_idx < model.layers.size(); ++layer_idx)
    {
        const auto& layer = model.layers[layer_idx];

        // --- Allocate and upload weight tensor to GPU ---
        dW[layer_idx].ensure_size(layer.in * layer.out);
        for (size_t i = 0; i < layer.out; ++i)
            for (size_t k = 0; k < layer.in; ++k)
                dW[layer_idx].host[i * layer.in + k] = layer.W(i, k);

#ifdef DEBUG
        printf("DEBUG Layer %zu host weights (first 2 rows, first 5 cols):\n", layer_idx);
        for (size_t i = 0; i < std::min<size_t>(2, layer.out); ++i)
        {
            for (size_t k = 0; k < std::min<size_t>(5, layer.in); ++k)
                printf("%f ", dW[layer_idx].host[i * layer.in + k]);
            printf("\n");
        }
#endif

        dW[layer_idx].upload();
        if (!dW[layer_idx].dev)
            throw std::runtime_error("dW GPU pointer null after upload");

        // --- Allocate and upload bias tensor to GPU ---
        dB[layer_idx].ensure_size(layer.out);
        for (size_t i = 0; i < layer.out; ++i)
            dB[layer_idx].host[i] = layer.b[i];

        dB[layer_idx].upload();
        if (!dB[layer_idx].dev)
            throw std::runtime_error("dB GPU pointer null after upload");
    }

    /**
     * @brief Perform a CUDA forward pass using pre-allocated GPU buffers
     * @param input_vec Input vector
     * @return Output vector from the last layer
     */
    auto cuda_forward = [&](const std::vector<float>& input_vec) -> std::vector<float>
        {
            if (model.layers.empty()) return {};

            // --- Upload input to GPU ---
            dX[0].ensure_size(input_vec.size());
            std::copy(input_vec.begin(), input_vec.end(), dX[0].host.begin());
            dX[0].upload();

#ifdef DEBUG
            for (size_t i = 0; i < std::min<size_t>(10, input_vec.size()); ++i)
                printf("Input[%zu] = %f\n", i, dX[0].host[i]);
#endif

            for (size_t i = 0; i < model.layers.size(); ++i)
            {
                const auto& layer = model.layers[i];

                // Ensure output buffer is allocated on GPU
                dY[i].ensure_size(layer.out);
                cudaMemset(dY[i].dev, 0, layer.out * sizeof(float));

                // --- CUDA forward for this layer ---
                fc_cuda_forward_reuse(dW[i].dev, dB[i].dev, dX[i].dev, dY[i].dev,
                    static_cast<int>(layer.in), static_cast<int>(layer.out));

                // Copy output to input of next layer (device-to-device)
                if (i + 1 < model.layers.size())
                {
                    const auto& next = model.layers[i + 1];
                    if (layer.out != next.in)
                        throw std::runtime_error("Layer size mismatch between layers " + std::to_string(i) + " and " + std::to_string(i + 1));

                    dX[i + 1].ensure_size(next.in);
                    cudaMemcpy(dX[i + 1].dev, dY[i].dev, layer.out * sizeof(float), cudaMemcpyDeviceToDevice);
                }

#ifdef DEBUG
                dY[i].download();
                printf("DEBUG Layer %zu output (first 10 values): ", i);
                for (size_t j = 0; j < std::min<size_t>(10, dY[i].size); ++j)
                    printf("%.6f ", dY[i].host[j]);
                printf("\n");
#endif
            }

            // Download final output
            dY.back().download();
            return dY.back().host;
        };

    // --- CPU vs CUDA correctness check ---
    std::vector<float> cpu_out = input;
    for (auto& L : model.layers)
        cpu_out = fc_forward_seq(L.W, L.b, cpu_out);

    std::vector<float> cuda_out = cuda_forward(input);

    auto vectors_equal = [](const std::vector<float>& a, const std::vector<float>& b)
        {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i)
                if (std::fabs(a[i] - b[i]) > 1e-4f) return false;
            return true;
        };

    std::cout << "CPU vs CUDA match? " << (vectors_equal(cpu_out, cuda_out) ? "YES" : "NO") << "\n";

    // === Performance benchmark ===
    const int warmup_runs = 5;
    const int repetitions = 200;

    // Warm-up CPU and GPU to stabilize performance
    for (int i = 0; i < warmup_runs; ++i)
    {
        std::vector<float> tmp = input;
        for (auto& L : model.layers) tmp = fc_forward_par(L.W, L.b, tmp);
        cuda_forward(input);
    }

    // --- Measure CPU time (parallel version) ---
    Timer timer;
    timer.reset();
    for (int i = 0; i < repetitions; ++i)
    {
        std::vector<float> tmp = input;
        for (auto& L : model.layers) tmp = fc_forward_par(L.W, L.b, tmp);
    }
    double cpu_ms = timer.elapsed_ms() / repetitions;

    // --- Measure CUDA time ---
    timer.reset();
    for (int i = 0; i < repetitions; ++i)
    {
        auto out = cuda_forward(input);
        (void)out;
    }
    double cuda_ms = timer.elapsed_ms() / repetitions;

    std::cout << "Average times (ms): CPU-par=" << cpu_ms << " | CUDA-reuse=" << cuda_ms << "\n";

    // --- Write benchmark results to CSV ---
    std::vector<std::string> header = { "input", "output", "t_cpu_ms", "t_cuda_ms" };
    std::vector<std::vector<double>> rows;
    rows.push_back({ (double)model.input_size, (double)model.output_size, cpu_ms, cuda_ms });

    write_csv(bench_path, header, rows);
    std::cout << "Results written to " << bench_path << "\n";

    return 0;
}
