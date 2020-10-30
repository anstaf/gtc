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

import enum
from typing import List

from eve import Bool, Int, Node, Str, StrEnum
from gtc import common


class Sid(Node):
    name: Str


class SparseField(Sid):
    connectivity: Str


class Temporary(Node):
    name: Str
    location_type: Str
    dtype: Str


class Composite(Node):
    name: Str
    items: List[Sid]


class Expr(Node):
    pass


class Literal(Expr):
    value: Str
    dtype: Str


class FieldAccess(Expr):
    name: Str
    location: Str


class VarAccess(Expr):
    name: Str


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


@enum.unique
class ReduceOperator(StrEnum):
    ADD = "sum"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


class NeighborReduction(Expr):
    op: ReduceOperator
    dtype: Str
    connectivity: Str
    max_neighbors: Int
    has_skip_values: Bool
    primary: Str
    secondary: Str
    body: Expr


class Stmt(Node):
    pass


class VarDecl(Stmt):
    name: Str
    init: Expr


class Assign(Stmt):
    left: FieldAccess
    right: Expr


class Kernel(Node):
    location_type: Str
    primary: Composite
    secondaries: List[Composite]
    body: List[Stmt]


class Computation(Node):
    name: Str
    connectivities: List[Str]
    params: List[Str]
    temporaries: List[Temporary]
    kernels: List[Kernel]
