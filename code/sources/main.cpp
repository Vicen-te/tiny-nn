

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== STANDARD HEADERS =====================
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;


// ===================== PROJECT HEADERS =====================
#include "Model.hpp"
#include "MNISTLoader.hpp"
#include "CpuLayer.hpp"
#include "GpuLayer.hpp"
#include "Timer.hpp"
#include "json.hpp"

using json = nlohmann::json;


// Undefine DEBUG to disable debug prints in this file
#ifdef DEBUG
#undef DEBUG
#endif

enum class Mode { TRAIN, INFERENCE, Unknown };
constinit int inference_index = 10;


int main(int argc, char** argv)
{
    std::filesystem::path root = std::filesystem::current_path().parent_path();
    std::cout << "[Info] Current directory: " << root << std::endl;

    if (argc < 2) 
    {
        std::cerr << "[Error] Usage: app <train|t> or <inference|i>\n";
        return 1;
    }

    std::string mode_str = argv[1];

    std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Map strings to enum values
    std::map<std::string, Mode> mode_map = 
    {
        {"t",           Mode::TRAIN},
        {"train",       Mode::TRAIN},
        {"i",           Mode::INFERENCE},
        {"inference",   Mode::INFERENCE},
    };

    Mode mode = Mode::Unknown;
    if (mode_map.find(mode_str) != mode_map.end())
        mode = mode_map[mode_str];

    // open config.json
    std::ifstream f(root / "config.json");
    if (!f.is_open()) throw std::runtime_error("Cannot open config file");
    json cfg; f >> cfg;

    fs::create_directories(root / cfg["paths"]["dataset"].get<std::string>());
    fs::create_directories(root / cfg["paths"]["models"].get<std::string>());
    fs::create_directories(root / cfg["paths"]["results"].get<std::string>());

    // === Dataset ===
    MNISTLoader loader;

    std::string_view loading = " Loading MNIST...\n";
    std::string_view loaded = " MNIST loaded successfully.\n";

    // Switch based on mode
    switch (mode)
    {
    case Mode::TRAIN:
    {
        std::cout << "[Train]" << loading;
        // Load training images and labels from MNIST dataset
        loader.load
        (
            root / cfg["paths"]["dataset"].get<std::string>() / "train-images.idx3-ubyte",
            root / cfg["paths"]["dataset"].get<std::string>() / "train-labels.idx1-ubyte"
        );
        std::cout << "[Train]" << loaded;

        // Retrieve data as vectors of floats
        std::vector<std::vector<float>> X = loader.get_images();
        std::vector<std::vector<float>> Y = loader.get_labels();

        Model<gpu::Layer> model;
        // Use this for random weight initialization
        //model.add_layer(784, 128, ActivationType::RELU);
        //model.add_layer(128, 10, ActivationType::SOFTMAX);
        //model.cpu_parallel = true; //< cpu only

        // Load pre-trained model from JSON for CPU/GPU consistency
        model.from_json
        (
            root / cfg["paths"]["models"].get<std::string>() /
            cfg["files"]["cpu-gpu"].get<std::string>()
        );

        // Start timer for training measurement
        Timer timer;

        // Train model with specified epochs, learning rate, and batch size
        model.train
        (
            X, Y, 
            cfg["training"]["epochs"].get<int>(), 
            cfg["training"]["lr"].get<float>(), 
            cfg["training"]["batch_size"].get<int>()
        );

        // Print training duration in seconds
        double training_time_ms = timer.elapsed_milliseconds();
        std::cout << "[Train] Training completed in " << training_time_ms / 1000.0
            << " seconds.\n";

        // Save the current model configuration and weights to a JSON file
        model.to_json
        (
            root / cfg["paths"]["models"].get<std::string>() /
            cfg["files"]["model"].get<std::string>()
        );

        [[fallthrough]];
    }
    case Mode::INFERENCE:
    {
        std::cout << "[Inference]" << loading;
        // Load test images and labels from MNIST dataset
        loader.load
        (
            root / cfg["paths"]["dataset"].get<std::string>() / "t10k-images.idx3-ubyte",
            root / cfg["paths"]["dataset"].get<std::string>() / "t10k-labels.idx1-ubyte"
        );
        std::cout << "[Inference]" << loaded;

        // Select a single sample image for inference
        std::vector<float> sample = loader.get_images()[inference_index];

        Model<cpu::Layer> model;

        // Load pre-trained model from JSON
        model.from_json
        (
            root / cfg["paths"]["models"].get<std::string>() /
            cfg["files"]["model"].get<std::string>()
        );

        // Run inference on selected sample
        std::vector<float> cuda_out = model.inference(sample);

        // Find maximum value and corresponding index in output
        int max_idx = 0;
        float max_val = cuda_out[0];
        for (size_t i = 1; i < cuda_out.size(); ++i)
        {
            if (cuda_out[i] > max_val)
            {
                max_val = cuda_out[i];
                max_idx = static_cast<int>(i);
            }
        }

        // Print maximum output value and its index
        std::cout << "[Inference] Maximum output value: " << max_val
            << " at index " << max_idx << std::endl;

        // Display ASCII preview of the sample
        std::cout << "[Inference] Sample preview:" << std::endl;
        loader.ascii_preview(inference_index);
        break;
    }
    case Mode::Unknown:
    default:
        std::cerr << "[Error] Unknown mode: " << mode_str << "\n";
        break;
    }

    // Wait for user input before exiting the program
    std::cout << "[Info] Press ENTER to exit...";
    std::cin.get();
	
    return 0;
}