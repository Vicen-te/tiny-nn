

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/MNISTLoader.hpp"


// ===================== STANDARD HEADERS =====================
#include <iostream>
#include <algorithm>
#include <stdexcept>


void MNISTLoader::load
(
    const std::filesystem::path& path_images, 
    const std::filesystem::path& path_labels
)
{
    std::ifstream ifs_images(path_images, std::ios::binary);
    std::ifstream ifs_labels(path_labels, std::ios::binary);

    if (!ifs_images.is_open() || !ifs_labels.is_open())
        throw std::runtime_error("Cannot open MNIST files");

    // Read MNIST file headers

    // Unused, but could be checked for format validation
    [[maybe_unused]] uint32_t magic_images = read_be_uint32(ifs_images); 
    n_images = read_be_uint32(ifs_images);
    n_rows = read_be_uint32(ifs_images);
    n_cols = read_be_uint32(ifs_images);

    [[maybe_unused]] uint32_t magic_labels = read_be_uint32(ifs_labels); //< Unused
    uint32_t n_labels = read_be_uint32(ifs_labels);

    if (n_images != n_labels)
        throw std::runtime_error("Images/labels count mismatch");

    // Allocate space for images and labels
    images.resize(n_images, std::vector<float>(n_rows * n_cols));
    labels.resize(n_labels, std::vector<float>(10, 0.0f));

    // Read image and label data
    for (size_t i = 0; i < n_images; ++i)
    {
        // Read and normalize pixel values to [0,1]
        for (size_t j = 0; j < n_rows * n_cols; ++j)
        {
            unsigned char pixel;
            ifs_images.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f;  // Normalize pixel to [0,1]
        }

        // Read label and convert to one-hot vector
        unsigned char label;
        ifs_labels.read(reinterpret_cast<char*>(&label), 1);
        labels[i][label] = 1.0f; // One-hot encoding
    }
}

void MNISTLoader::ascii_preview(size_t index) const
{
    if (index >= n_images) throw std::out_of_range("Image index out of range");

    const auto& img = images[index];

    // Define ASCII gradient for intensity levels
    const char ascii_chars[] = { ' ', '.', '*', '#', '@' };
    const size_t n_levels = sizeof(ascii_chars) / sizeof(ascii_chars[0]);

    // Iterate over each row and column to print ASCII representation
    for (size_t r = 0; r < n_rows; ++r)
    {
        for (size_t c = 0; c < n_cols; ++c)
        {
            float px_norm = img[r * n_cols + c]; //< Pixel normalized to [0,1]

            // Map pixel [0,1] to ascii index
            size_t idx = std::min(static_cast<size_t>(px_norm * n_levels), n_levels - 1);
            std::cout << ascii_chars[idx];
        }
        std::cout << "\n";
    }
}
