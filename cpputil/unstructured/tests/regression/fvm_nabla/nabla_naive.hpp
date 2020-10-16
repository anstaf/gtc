#pragma once

#include <gridtools/common/defs.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

#include <gridtools/next/domain.hpp>
#include <gridtools/next/tmp_storage.hpp>
#include <gridtools/next/unstructured.hpp>

namespace nabla_impl_ {

    using namespace gridtools;
    using namespace next;

    template <class Dim, class Sid>
    auto max_neighbors(Sid const &sid) {
        static_assert(has_key<sid::upper_bounds_type<Sid>, Dim>());
        return at_key<Dim>(sid::get_upper_bounds(sid));
    }

    struct v2e_dim;
    struct e2v_dim;
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

    template <class V2E, class E2V>
    auto nabla(domain d, V2E &&v2e, E2V &&e2v) {
        static_assert(is_sid<decltype(std::forward<V2E>(v2e)(storage::cpu_ifirst()))>());
        static_assert(is_sid<decltype(std::forward<E2V>(e2v)(storage::cpu_ifirst()))>());
        return [d = std::move(d),
                   v2e = sid::rename_dimensions<integral_constant<int_t, 1>, v2e_dim>(
                       std::forward<V2E>(v2e)(storage::cpu_ifirst())),
                   e2v = sid::rename_dimensions<integral_constant<int_t, 1>, e2v_dim>(
                       std::forward<E2V>(e2v)(storage::cpu_ifirst()))](auto &&S_MXX,
                   auto &&S_MYY,
                   auto &&pp,
                   auto &&pnabla_MXX,
                   auto &&pnabla_MYY,
                   auto &&vol,
                   auto &&sign) {
            static_assert(is_sid<decltype(S_MXX)>());
            static_assert(is_sid<decltype(S_MYY)>());
            static_assert(is_sid<decltype(pp)>());
            static_assert(is_sid<decltype(pnabla_MXX)>());
            static_assert(is_sid<decltype(pnabla_MYY)>());
            static_assert(is_sid<decltype(vol)>());
            static_assert(is_sid<decltype(sign)>());

            // allocate temporary field storage
            auto alloc = sid::make_cached_allocator(&std::make_unique<char[]>);
            auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
            auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);

            {
                auto fields = tuple_util::make<
                    sid::composite::keys<e2v_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>::values>(
                    e2v, S_MXX, S_MYY, zavgS_MXX, zavgS_MYY);
                sid::make_loop<dim::h>(d.edge)(
                    [&, pp = sid::get_origin(pp)(), pp_stride = at_key<dim::h>(sid::get_strides(pp))](
                        auto &ptr, auto const &strides) {
                        double acc = 0;
                        sid::make_loop<e2v_dim>(max_neighbors<e2v_dim>(e2v))([&](auto const &ptr, auto &&) {
                            acc += *sid::shifted(pp, pp_stride, *at_key<e2v_tag>(ptr));
                        })(ptr, strides);
                        acc *= 0.5;
                        *at_key<zavgS_MXX_tag>(ptr) = *at_key<S_MXX_tag>(ptr) * acc;
                        *at_key<zavgS_MYY_tag>(ptr) = *at_key<S_MYY_tag>(ptr) * acc;
                    })(sid::get_origin(fields)(), sid::get_strides(fields));
            }
            {
                auto fields =
                    tuple_util::make<sid::composite::keys<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag>::values>(
                        v2e, pnabla_MXX, pnabla_MYY, sid::rename_dimensions<dim::n, v2e_dim>(sign));
                sid::make_loop<dim::h>(d.vertex)([&,
                                                     zavgS_MXX = sid::get_origin(zavgS_MXX)(),
                                                     zavgS_MXX_stride = at_key<dim::h>(sid::get_strides(zavgS_MXX)),
                                                     zavgS_MYY = sid::get_origin(zavgS_MYY)(),
                                                     zavgS_MYY_stride = at_key<dim::h>(sid::get_strides(zavgS_MYY))](
                                                     auto &ptr, auto const &strides) {
                    double acc = 0;
                    sid::make_loop<v2e_dim>(max_neighbors<v2e_dim>(v2e))([&](auto const &ptr, auto &&) {
                        auto i = *at_key<v2e_tag>(ptr);
                        if (i < 0)
                            return;
                        acc += *sid::shifted(zavgS_MXX, zavgS_MXX_stride, i) * *at_key<sign_tag>(ptr);
                    })(ptr, strides);
                    *at_key<pnabla_MXX_tag>(ptr) = acc;
                    acc = 0;
                    sid::make_loop<v2e_dim>(max_neighbors<v2e_dim>(v2e))([&](auto const &ptr, auto &&) {
                        auto i = *at_key<v2e_tag>(ptr);
                        if (i < 0)
                            return;
                        acc += *sid::shifted(zavgS_MYY, zavgS_MYY_stride, i) * *at_key<sign_tag>(ptr);
                    })(ptr, strides);
                    *at_key<pnabla_MYY_tag>(ptr) = acc;
                })(sid::get_origin(fields)(), sid::get_strides(fields));
            }
            {
                auto fields = tuple_util::make<sid::composite::keys<pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>::values>(
                    pnabla_MXX, pnabla_MYY, vol);
                sid::make_loop<dim::h>(d.vertex)([](auto &ptr, auto &&stride) {
                    *at_key<pnabla_MXX_tag>(ptr) /= *at_key<vol_tag>(ptr);
                    *at_key<pnabla_MYY_tag>(ptr) /= *at_key<vol_tag>(ptr);
                })(sid::get_origin(fields)(), sid::get_strides(fields));
            }
        };
    }
} // namespace nabla_impl_
using nabla_impl_::nabla;
