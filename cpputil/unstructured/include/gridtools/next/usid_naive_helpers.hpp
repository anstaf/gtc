#pragma once

#include <memory>
#include <tuple>

#include <gridtools/common/hymap.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

#include "unstructured.hpp"

namespace gridtools::next::naive {
    using traits_t = storage::cpu_ifirst;

    inline auto make_allocator() { return sid::make_cached_allocator(&std::make_unique<char[]>); }

    template <class Size, class Body, class Sid, class... Sids>
    void call_kernel(Size size, Body &&body, Sid &&fields, Sids &&...neighbor_fields) {
        sid::make_loop<dim::h>(size)(
            [&body,
                params = std::tuple_cat(std::make_tuple(sid::get_origin(neighbor_fields)(),
                    sid::get_stride<dim::h>(sid::get_strides(neighbor_fields)))...)](auto &ptr, auto const &strides) {
                std::apply(body, std::tuple_cat(std::forward_as_tuple(ptr, strides), params));
            })(sid::get_origin(fields)(), sid::get_strides(fields));
    }

    template <class Tag, class Ptr>
    decltype(auto) field(Ptr const &ptr) {
        return *at_key<Tag>(ptr);
    }
} // namespace gridtools::next::naive
