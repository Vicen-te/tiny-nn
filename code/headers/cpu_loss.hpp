

// Copyright (c) 2025 Vicente Brisa Saez
// Github: Vicen-te
// License: MIT


#pragma once


// ===================== STANDARD HEADERS =====================
#include <span>
#include <algorithm>
#include <cmath>


namespace cpu::loss
{

    constinit float eps = 1e-7f; //< Small value to avoid log(0)

    /**
     * @brief Compute the cross-entropy loss for a single sample.
     *
     * This function assumes `y_true` is one-hot encoded and `y_pred` contains
     * predicted probabilities (after softmax). Uses epsilon clipping to avoid
     * log(0) issues.
     *
     * @param y_true Vector of true labels (one-hot).
     * @param y_pred Vector of predicted probabilities.
     * @return Cross-entropy loss as a float.
     */
    [[nodiscard]] inline float cross_entropy
    (
        std::span<const float> y_true,
        std::span<const float> y_pred
    )
    {
        float loss = 0.0f;
        for (size_t i = 0; i < y_true.size(); ++i)
            loss -= y_true[i] * std::log(std::clamp(y_pred[i], eps, 1.0f));

        return loss;
    }

}
