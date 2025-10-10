

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/Model.hpp"


// ===================== STANDARD LIBRARIES =====================

#include <stdexcept>
#include <fstream>


// ===================== CUSTOM HEADERS =====================
#include "../headers/Layer.hpp"
#include "json.hpp"

using json = nlohmann::json;


void Model::save_json(const std::string& path) const
{
    json file;
    file["input_size"] = input_size;
    file["output_size"] = output_size;
    file["layers"] = json::array();

    // Serialize each layer
    for (const auto& L : layers)
        file["layers"].push_back(L.to_json());

    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    ofs << file.dump(4); //< Pretty-print JSON with 4-space indentation
}

void Model::load_json(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open file for reading: " + path);

    json file;
    ifs >> file;

    input_size = file["input_size"];
    output_size = file["output_size"];

    layers.clear(); //< Remove existing layers

    // Deserialize each layer
    for (auto& jl : file["layers"])
    {
        Layer L;
        L.from_json(jl);
        layers.push_back(L);
    }
}
