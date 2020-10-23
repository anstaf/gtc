#pragma once

#include <type_traits>

#include <gridtools/common/host_device.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/utility.hpp>
#include <gridtools/meta/id.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/contiguous.hpp>
#include <gridtools/sid/loop.hpp>

#include "unstructured.hpp"

namespace gridtools::next {
    template <class T, class Alloc, class HSize, class KSize>
    auto make_simple_tmp_storage(HSize h_size, KSize k_size, Alloc &alloc) {
        return sid::make_contiguous<T>(alloc, hymap::keys<dim::h, dim::k>::values<HSize, KSize>(h_size, k_size));
    }

    template <class Tag, class F, class Init, class N, class G, class HasSkipValues>
    GT_FUNCTION auto fold_neighbors(F f, Init init, N n, G g, HasSkipValues) {
        return [=, loop = sid::make_loop<Tag>(n)](
                   auto &&ptr, auto const &strides, auto &&neighbor_ptr, auto &&neighbor_stride) {
            using acc_t = std::decay_t<decltype(g(ptr, neighbor_ptr))>;
            acc_t acc = init(meta::lazy::id<acc_t>());
            loop([&](auto const &ptr, auto &&) {
                auto i = *at_key<Tag>(ptr);
                if constexpr (HasSkipValues::value)
                    if (i < 0)
                        return;
                acc = f(acc, g(ptr, sid::shifted(neighbor_ptr, neighbor_stride, i)));
            })(wstd::forward<decltype(ptr)>(ptr), strides);
            return acc;
        };
    }

    template <class Tag>
    struct sum_neighbors_f {
        template <class N, class F, class HasSkip = std::true_type>
        GT_FUNCTION auto operator()(N n, F f, HasSkip has_skip = {}) const {
            return fold_neighbors<Tag>([](auto x, auto y) { return x + y; },
                [](auto z) -> typename decltype(z)::type { return {}; },
                n,
                f,
                has_skip);
        }
    };

    template <class Tag>
    constexpr sum_neighbors_f<Tag> sum_neighbors = {};

    template <class... Keys>
    constexpr auto make_composite = [](auto &&...sids) {
        return tuple_util::make<sid::composite::keys<Keys...>::template values>(std::forward<decltype(sids)>(sids)...);
    };
} // namespace gridtools::next