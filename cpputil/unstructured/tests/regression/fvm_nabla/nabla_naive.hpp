#pragma once

#include <utility>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/next/domain.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/next/usid_helpers.hpp>
#include <gridtools/next/usid_naive_helpers.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/rename_dimensions.hpp>

namespace gridtools::next::naive::nabla_impl_ {
    using namespace literals;
    struct v2e_tag;
    struct e2v_tag;
    struct S_MXX_tag;
    struct S_MYY_tag;
    struct zavgS_MXX_tag;
    struct zavgS_MYY_tag;
    struct pnabla_MXX_tag;
    struct pnabla_MYY_tag;
    struct vol_tag;
    struct sign_tag;
    struct pp_tag;
    constexpr auto nabla = [](domain d, auto &&v2e, auto &&e2v) {
        static_assert(is_sid<decltype(v2e(traits_t()))>());
        static_assert(is_sid<decltype(e2v(traits_t()))>());
        return
            [d = std::move(d),
                v2e = sid::rename_dimensions<dim::n, v2e_tag>(std::forward<decltype(v2e)>(v2e)(traits_t())),
                e2v = sid::rename_dimensions<dim::n, e2v_tag>(std::forward<decltype(e2v)>(e2v)(traits_t()))](
                auto &&S_MXX, auto &&S_MYY, auto &&pp, auto &&pnabla_MXX, auto &&pnabla_MYY, auto &&vol, auto &&sign) {
                static_assert(is_sid<decltype(S_MXX)>());
                static_assert(is_sid<decltype(S_MYY)>());
                static_assert(is_sid<decltype(pp)>());
                static_assert(is_sid<decltype(pnabla_MXX)>());
                static_assert(is_sid<decltype(pnabla_MYY)>());
                static_assert(is_sid<decltype(vol)>());
                static_assert(is_sid<decltype(sign)>());
                auto alloc = make_allocator();
                auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
                auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
                call_kernel(
                    d.edge,
                    [](auto &ptr, auto const &strides, auto const &neighbor_ptr, auto const &neighbor_stride) {
                        auto zavg = 0.5 * sum_neighbors<e2v_tag>(2_c, [](auto &&, auto &&n) {
                            return field<pp_tag>(n);
                        })(ptr, strides, neighbor_ptr, neighbor_stride);
                        field<zavgS_MXX_tag>(ptr) = field<S_MXX_tag>(ptr) * zavg;
                        field<zavgS_MYY_tag>(ptr) = field<S_MYY_tag>(ptr) * zavg;
                    },
                    make_composite<e2v_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>(
                        e2v, S_MXX, S_MYY, zavgS_MXX, zavgS_MYY),
                    make_composite<pp_tag>(pp));
                call_kernel(
                    d.vertex,
                    [](auto &ptr, auto const &strides, auto const &neighbor_ptr, auto const &neighbor_stride) {
                        field<pnabla_MXX_tag>(ptr) = sum_neighbors<v2e_tag>(7_c, [](auto &&ptr, auto &&neighbor_ptr) {
                            return field<zavgS_MXX_tag>(neighbor_ptr) * field<sign_tag>(ptr);
                        })(ptr, strides, neighbor_ptr, neighbor_stride);
                        field<pnabla_MYY_tag>(ptr) = sum_neighbors<v2e_tag>(7_c, [](auto &&ptr, auto &&neighbor_ptr) {
                            return field<zavgS_MYY_tag>(neighbor_ptr) * field<sign_tag>(ptr);
                        })(ptr, strides, neighbor_ptr, neighbor_stride);
                    },
                    make_composite<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag>(
                        v2e, pnabla_MXX, pnabla_MYY, sid::rename_dimensions<dim::s, v2e_tag>(sign)),
                    make_composite<zavgS_MXX_tag, zavgS_MYY_tag>(zavgS_MXX, zavgS_MYY));
                call_kernel(
                    d.vertex,
                    [](auto &ptr, auto const &strides) {
                        field<pnabla_MXX_tag>(ptr) = field<pnabla_MXX_tag>(ptr) / field<vol_tag>(ptr);
                        field<pnabla_MYY_tag>(ptr) = field<pnabla_MYY_tag>(ptr) / field<vol_tag>(ptr);
                    },
                    make_composite<pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>(pnabla_MXX, pnabla_MYY, vol));
            };
    };
} // namespace gridtools::next::naive::nabla_impl_
using gridtools::next::naive::nabla_impl_::nabla;
