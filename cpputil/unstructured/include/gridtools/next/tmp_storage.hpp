#pragma once

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/contiguous.hpp>

#include "unstructured.hpp"

namespace gridtools {
    namespace next {
        template <class T, class Alloc, class HSize, class KSize>
        auto make_simple_tmp_storage(HSize h_size, KSize k_size, Alloc &alloc) {
            return sid::make_contiguous<T>(
                alloc, tuple_util::make<hymap::keys<dim::h, dim::k>::values>(h_size, k_size));
        }
    } // namespace next
} // namespace gridtools
