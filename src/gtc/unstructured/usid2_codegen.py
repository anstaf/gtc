# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc.unstructured import usid2


class _KernelCallGenerator(codegen.TemplatedGenerator):
    Sid = as_fmt('{name}')
    SparseField = as_fmt('sid::rename_dimensions<dim::s, {connectivity}_tag>({name})')
    Composite = as_mako(
        "sid::composite::make<${', '.join(f'{i.name}_tag' for i in _this_node.items)}>(${', '.join(items)})")
    Kernel = as_mako(
        "call_kernel<${id_}>(d.${location_type}${''.join(f', {c}' for c in [primary] + secondaries)});")


class _Generator(codegen.TemplatedGenerator):
    def visit_Computation(self, node: usid2.Computation, **kwargs):
        return self.generic_visit(node, kernel_calls=[_KernelCallGenerator.apply(k) for k in node.kernels], **kwargs)

    Literal = as_mako('${dtype}{${value}}')
    BinaryOp = as_fmt('({left} {op} {right})')
    FieldAccess = as_fmt('field<{name}_tag>({location})')
    VarAccess = as_fmt('{name}')
    NeighborReduction = as_mako(
        '${op}_neighbors<${dtype}, ${connectivity}_tag, ${max_neighbors}, ${has_skip_values.lower()}>' +
        '([](auto &&${primary}, auto &&${secondary}) { return ${body}; }, ${primary}, strides, ${secondary})')
    VarDecl = as_fmt('auto&& {name} = {init};')
    Assign = as_fmt('{left} = {right};')
    Kernel = as_mako('''struct ${id_} {
    GT_FUNCTION auto operator()() const {
        return [](auto && ${_this_node.primary.name}, auto &&strides${''.join(f', auto&& {s.name}' for s in _this_node.secondaries)}) {
            ${'\\n            '.join(body)}
        };
    }
};''')
    Temporary = as_fmt(
        'auto {name} = make_simple_tmp_storage<{dtype}>(d.{location_type}, d.k, alloc);')
    Computation = as_mako("""#pragma once
#include <gridtools/usid/${ backend }_helpers.hpp>
namespace gridtools::usid::${ backend }::${ name }_impl_ {
${'\\n'.join(f'struct {f}_tag;' for f in connectivities + params + [t.name for t in _this_node.temporaries])}
${'\\n'.join(kernels)}
inline constexpr auto ${ name } = [](domain d${ ''.join(f', auto&& {c}' for c in connectivities) }) {
    ${'\\n    '.join(f'static_assert(is_sid<decltype({c}(traits_t()))>());' for c in connectivities)}
    return[d = std::move(d)${ ''.join(f', {c} = sid::rename_dimensions<dim::n, {c}_tag>(std::forward<decltype({c})>({c})(traits_t()))' for c in connectivities) }]
    (${ ', '.join(f'auto&& {p}' for p in params)}) {
        ${'\\n        '.join(f'static_assert(is_sid<decltype({p})>());' for p in params)}
% if len(temporaries) > 0:
        auto alloc = make_allocator();
        ${'\\n        '.join(temporaries)}
% endif
        ${'\\n        '.join(kernel_calls)}
    };
};
}
using gridtools::usid::${ backend }::${ name }_impl_::${ name };
""")


def _impl(backend):
    return lambda src: codegen.format_source('cpp', _Generator.apply(src, backend=backend), style='LLVM')


naive = _impl('naive')
gpu = _impl('gpu')
