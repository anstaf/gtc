#pragma once

#ifndef __CUDACC__
#error "Tried to compile CUDA code with a regular C++ compiler."
#endif

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/gpu.hpp>

#include <gridtools/next/cuda_util.hpp>
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

    template <class Ptr, class Loop, class Strides, class Pp, class PPStride>
    __global__ void nabla_edge_1(int e_size, Loop loop, Ptr ptr_holder, Strides strides, Pp pp, PPStride pp_stride) {
        {
            auto idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= e_size)
                return;
            auto ptr = ptr_holder();
            sid::shift(ptr, device::at_key<dim::h>(strides), idx);
            double acc = 0;
            loop([&acc, pp_stride, pp = pp()](auto const &ptr, auto &&) {
                acc += *sid::shifted(pp, pp_stride, *device::at_key<e2v_tag>(ptr));
            })(ptr, strides);
            acc *= 0.5;
            *device::at_key<zavgS_MXX_tag>(ptr) = *device::at_key<S_MXX_tag>(ptr) * acc;
            *device::at_key<zavgS_MYY_tag>(ptr) = *device::at_key<S_MYY_tag>(ptr) * acc;
        }
    }

    template <class Ptr,
        class Loop,
        class Strides,
        class ZavgS_MXX,
        class ZavgS_MXXStride,
        class ZavgS_MYY,
        class ZavgS_MYYStride>
    __global__ void nabla_vertex_2(int v_size,
        Loop loop,
        Ptr ptr_holder,
        Strides strides,
        ZavgS_MXX zavgS_MXX,
        ZavgS_MXXStride zavgS_MXX_stride,
        ZavgS_MYY zavgS_MYY,
        ZavgS_MYYStride zavgS_MYY_stride) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= v_size)
            return;
        auto ptr = ptr_holder();
        sid::shift(ptr, gridtools::device::at_key<dim::h>(strides), idx);
        double acc = 0;
        loop([&acc, zavgS_MXX = zavgS_MXX(), zavgS_MXX_stride](auto const &ptr, auto &&) {
            auto i = *device::at_key<v2e_tag>(ptr);
            if (i < 0)
                return;
            acc += *sid::shifted(zavgS_MXX, zavgS_MXX_stride, i) * *device::at_key<sign_tag>(ptr);
        })(ptr, strides);
        *at_key<pnabla_MXX_tag>(ptr) = acc;
        acc = 0;
        loop([&acc, zavgS_MYY = zavgS_MYY(), zavgS_MYY_stride](auto const &ptr, auto &&) {
            auto i = *device::at_key<v2e_tag>(ptr);
            if (i < 0)
                return;
            acc += *sid::shifted(zavgS_MYY, zavgS_MYY_stride, i) * *device::at_key<sign_tag>(ptr);
        })(ptr, strides);
        *at_key<pnabla_MYY_tag>(ptr) = acc;
    }

    template <class Ptr, class Strides>
    __global__ void nabla_vertex_4(int v_size, Ptr ptr_holder, Strides strides) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= v_size)
            return;
        auto ptr = ptr_holder();
        sid::shift(ptr, device::at_key<dim::h>(strides), idx);
        *device::at_key<pnabla_MXX_tag>(ptr) /= *device::at_key<vol_tag>(ptr);
        *device::at_key<pnabla_MYY_tag>(ptr) /= *device::at_key<vol_tag>(ptr);
    }

    template <class V2E, class E2V>
    auto nabla(domain d, V2E &&v2e, E2V &&e2v) {
        static_assert(is_sid<decltype(std::forward<V2E>(v2e)(storage::gpu()))>());
        static_assert(is_sid<decltype(std::forward<E2V>(e2v)(storage::gpu()))>());
        return [d = std::move(d),
                   v2e = sid::rename_dimensions<integral_constant<int_t, 1>, v2e_dim>(
                       std::forward<V2E>(v2e)(storage::gpu())),
                   e2v = sid::rename_dimensions<integral_constant<int_t, 1>, e2v_dim>(
                       std::forward<E2V>(e2v)(storage::gpu()))](auto &&S_MXX,
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
            auto alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char[]>);
            auto zavgS_MXX = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
            auto zavgS_MYY = make_simple_tmp_storage<double>(d.edge, d.k, alloc);
            {
                auto fields = tuple_util::make<
                    sid::composite::keys<e2v_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>::values>(
                    e2v, S_MXX, S_MYY, zavgS_MXX, zavgS_MYY);
                auto [blocks, threads_per_block] = cuda_setup(d.edge);
                nabla_edge_1<<<blocks, threads_per_block>>>(d.edge,
                    sid::make_loop<e2v_dim>(max_neighbors<e2v_dim>(e2v)),
                    sid::get_origin(fields),
                    sid::get_strides(fields),
                    sid::get_origin(pp),
                    at_key<dim::h>(sid::get_strides(pp)));
                GT_CUDA_CHECK(cudaGetLastError());
            }
            {
                auto fields =
                    tuple_util::make<sid::composite::keys<v2e_tag, pnabla_MXX_tag, pnabla_MYY_tag, sign_tag>::values>(
                        v2e, pnabla_MXX, pnabla_MYY, sid::rename_dimensions<dim::n, v2e_dim>(sign));
                auto [blocks, threads_per_block] = cuda_setup(d.vertex);
                nabla_vertex_2<<<blocks, threads_per_block>>>(d.vertex,
                    sid::make_loop<v2e_dim>(max_neighbors<v2e_dim>(v2e)),
                    sid::get_origin(fields),
                    sid::get_strides(fields),
                    sid::get_origin(zavgS_MXX),
                    at_key<dim::h>(sid::get_strides(zavgS_MXX)),
                    sid::get_origin(zavgS_MYY),
                    at_key<dim::h>(sid::get_strides(zavgS_MYY)));
                GT_CUDA_CHECK(cudaGetLastError());
            }
            {
                auto fields = tuple_util::make<sid::composite::keys<pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>::values>(
                    pnabla_MXX, pnabla_MYY, vol);
                auto [blocks, threads_per_block] = cuda_setup(d.vertex);
                nabla_vertex_4<<<blocks, threads_per_block>>>(
                    d.vertex, sid::get_origin(fields), sid::get_strides(fields));
                GT_CUDA_CHECK(cudaGetLastError());
            }
        };
    }
} // namespace nabla_impl_
using nabla_impl_::nabla;
