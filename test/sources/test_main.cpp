

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


// ===================== STANDARD HEADERS =====================
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;


// ===================== LIBRARY HEADERS =====================
#include "Model.hpp"
#include "CpuLayer.hpp"
#include "MNISTLoader.hpp"
#include "json.hpp"

using json = nlohmann::json;


// ===================== PROJECT HEADERS =====================
#include "../headers/benchmark.hpp"


enum class Mode { COMPARE, VERIFY, Unknown };
constinit int inference_index = 10;


int main(int argc, char** argv)
{
    try
    {
        // Determine root directory (two levels up from current path)
        std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path();
        std::cout << "[Info] Current directory: " << root << std::endl;

        // Check command-line arguments
        if (argc < 2)
        {
            std::cerr << "[Error] Usage: app <compare|c> or <verify|v>\n";
            return 1;
        }

        // Convert input mode to lowercase
        std::string mode_str = argv[1];
        std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        // Map strings to enum values
        std::map<std::string, Mode> mode_map =
        {
            {"c",       Mode::COMPARE},
            {"compare", Mode::COMPARE},
            {"v",       Mode::VERIFY},
            {"verify",  Mode::VERIFY},
        };

        Mode mode = Mode::Unknown;
        if (mode_map.find(mode_str) != mode_map.end())
            mode = mode_map[mode_str];

        // Open config.json
        std::ifstream f(root / "config.json");
        if (!f.is_open()) throw std::runtime_error("Cannot open config file");
        json cfg; f >> cfg;

        // Ensure required directories exist
        fs::create_directories(root / cfg["paths"]["dataset"].get<std::string>());
        fs::create_directories(root / cfg["paths"]["models"].get<std::string>());
        fs::create_directories(root / cfg["paths"]["results"].get<std::string>());

        // Switch based on mode
        switch (mode)
        {
        case Mode::VERIFY:
        {
            std::cout << "[Verify] Loading MNIST...\n";

            // Load MNIST test dataset
            MNISTLoader loader;
            loader.load(
                root / cfg["paths"]["dataset"].get<std::string>() / "t10k-images.idx3-ubyte",
                root / cfg["paths"]["dataset"].get<std::string>() / "t10k-labels.idx1-ubyte"
            );
            std::cout << "[Verify] MNIST loaded successfully.\n";

            // Load pre-trained model
            Model<cpu::Layer> model;
            model.from_json(
                root / cfg["paths"]["models"].get<std::string>() /
                cfg["files"]["model"].get<std::string>()
            );

            // Get sample and expected label
            const std::vector<float>& sample = loader.get_images()[inference_index];
            const std::vector<float>& expected_label = loader.get_labels()[inference_index];

            // Run inference
            std::vector<float> output = model.inference(sample);

            // Verify inference result
            std::cout << "[Verify] Sample index: " << inference_index << std::endl << std::endl;
            bool correct = benchmark::verify_inference(output, expected_label);
            std::cout << "[Verify] Inference " << (correct ? "successful\n" : "mismatch\n");
            break;
        }
        case Mode::COMPARE:
        {
            std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path();
            std::cout << "[Compare] Running from: " << root << "\n";

            // Open config.json file
            std::ifstream f(root / "config.json");
            if (!f.is_open()) throw std::runtime_error("Cannot open config file");
            json cfg;
            f >> cfg;

            // Ensure results directory exists
            std::filesystem::create_directories(root / cfg["paths"]["results"].get<std::string>());

            // Run benchmark to compare CPU and GPU models
            benchmark::compare_models(
                root / cfg["paths"]["models"].get<std::string>() /
                cfg["files"]["model"].get<std::string>(),
                root / cfg["paths"]["results"].get<std::string>() /
                cfg["files"]["benchmark"].get<std::string>()
            );

            std::cout << "[Compare] CPU–GPU comparison completed successfully\n";
            break;
        }
        case Mode::Unknown:
        default:
            std::cerr << "[Error] Unknown mode: " << mode_str << "\n";
            break;
        }
    }
    catch (const std::exception& e)
    {
        // Catch and report exceptions
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
