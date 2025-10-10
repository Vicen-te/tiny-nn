

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD LIBRARIES =====================
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <cmath>


/**
 * @brief Simple loader for the MNIST handwritten digits dataset.
 *
 * Handles reading the binary ubyte format, normalizing image pixels to [0,1],
 * converting labels to one-hot encoding, and providing basic inspection utilities.
 */
class MNISTLoader
{

private:

    std::vector<std::vector<float>> images; ///< Normalized MNIST images
    std::vector<std::vector<float>> labels; ///< One-hot encoded labels

    size_t n_images = 0;  ///< Total number of images
    size_t n_rows = 0;    ///< Number of rows per image
    size_t n_cols = 0;    ///< Number of columns per image

    /**
     * @brief Reads a 32-bit unsigned integer from a binary stream in big-endian format.
     *
     * MNIST files are stored in big-endian, so we convert to little-endian for proper usage on x86 CPUs.
     *
     * @param ifs Input file stream (binary)
     * @return Converted 32-bit unsigned integer
     */
    static inline uint32_t read_be_uint32(std::ifstream& ifs)
    {
        uint32_t x = 0;
        ifs.read(reinterpret_cast<char*>(&x), 4);

        // Convert big-endian to little-endian
        return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24);
    }


public:

    MNISTLoader() = default;


    // ===================== ACCESSORS =====================

    /**
     * @brief Get a reference to the loaded images
     * @return Const reference to 2D vector of normalized image pixels
     */
    inline const std::vector<std::vector<float>>& get_images() const { return images; }

    /**
     * @brief Get a reference to the loaded labels
     * @return Const reference to 2D vector of one-hot encoded labels
     */
    inline const std::vector<std::vector<float>>& get_labels() const { return labels; }

    /**
     * @brief Return the number of images
     * @return Number of loaded images
     */
    inline const size_t size() const { return n_images; }

    /**
     * @brief Return the number of rows in each image
     * @return Image height (rows)
     */
    inline const size_t rows() const { return n_rows; }

    /**
     * @brief Return the number of columns in each image
     * @return Image width (columns)
     */
    inline const size_t cols() const { return n_cols; }


    // ===================== DATASET MANAGEMENT =====================

    /**
     * @brief Load MNIST dataset from the given image and label files.
     *
     * This method normalizes image pixels to [0,1] and converts labels to one-hot vectors.
     * It also performs basic checks on file validity and image/label consistency.
     *
     * @param path_images Path to the MNIST images file (ubyte format)
     * @param path_labels Path to the MNIST labels file (ubyte format)
     */
    void load(const std::string& path_images, const std::string& path_labels);

    /**
     * @brief Simple ASCII visualization of a single MNIST image.
     *
     * Useful for debugging or quick inspection of the dataset.
     * Pixel intensity is mapped to symbols:
     *   @ -> darkest, # -> dark, * -> medium, . -> light, ' ' -> near zero
     *
     * @param index Index of the image to display
     */
    void ascii_preview(size_t index = 0) const;
};
