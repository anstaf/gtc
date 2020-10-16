#pragma once

#include <gridtools/common/defs.hpp>
#include <gridtools/common/integral_constant.hpp>

namespace gridtools::next::dim {
    using horizontal = integral_constant<int_t, 0>;
    using vertical = integral_constant<int_t, 1>;
    using neighbor = integral_constant<int_t, 2>;

    using h = horizontal;
    using k = vertical;
    using n = neighbor;
} // namespace gridtools::next::dim
