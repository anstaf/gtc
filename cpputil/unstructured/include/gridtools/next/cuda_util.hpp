#pragma once

#include <tuple>

namespace gridtools::next {
    inline auto cuda_setup(int N) {
        int threads_per_block = 32;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        return std::make_tuple(blocks, threads_per_block);
    }
} // namespace gridtools::next
