

// Copyright (c) 2025 Vicente Brisa Saez
// GitHub: Vicen-te
// License: MIT


// ===================== MAIN HEADER =====================
#include "../headers/ActivationType.hpp"


// ===================== STANDARD HEADERS =====================
#include <ranges>


namespace activation
{

    std::vector<float> softmax(const std::vector<float>& z)
    {
        std::vector<float> res = z;

        // Find the maximum element for numerical stability
        float max_val = *std::ranges::max_element(res);

        // Compute exponential and sum
        float sum = 0.0f;
        for (auto& score : res)
            sum += (score = std::exp(score - max_val));

        // Normalize
        for (auto& score : res)
            score /= sum;

        return res;
    }

}
