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


from typing import List, Optional, Set, Tuple, Union

from devtools import debug
from pydantic import root_validator, validator

import eve
from eve import Node, Str
from gtc import common


class Expr(Node):
    location_type: common.LocationType


class Stmt(Node):
    location_type: common.LocationType


class NeighborChain(Node):
    elements: Tuple[common.LocationType, ...]

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.elements)

    def __eq__(self, other):
        return self.elements == other.elements

    @validator("elements")
    def not_empty(cls, elements):
        if len(elements) < 1:
            raise ValueError("NeighborChain must contain at least one locations")
        return elements


class FieldAccess(Expr):
    name: Str  # symbol ref to SidCompositeEntry
    sid: Str  # symbol ref


class VarDecl(Stmt):
    name: Str
    init: Expr
    vtype: common.DataType


class Literal(Expr):
    value: Union[common.BuiltInLiteral, Str]
    vtype: common.DataType


class VarAccess(Expr):
    name: Str  # via symbol table
    dummy: Optional[
        Str
    ]  # to distinguish from FieldAccess, see https://github.com/eth-cscs/eve_toolchain/issues/34


class AssignStmt(Stmt):
    left: Union[FieldAccess, VarAccess]
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # if values["left"].location_type != values["right"].location_type:
        #     raise ValueError("Location type mismatch")

        # if "location_type" not in values:
        #     values["location_type"] = values["left"].location_type
        # elif values["left"] != values["location_type"]:
        #     raise ValueError("Location type mismatch")
        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

        return values


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")

        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

        return values


class SidCompositeEntry(Node):
    name: Str  # symbol decl (TODO ensure field exists via symbol table)

    @property
    def tag_name(self):
        return self.name + "_tag"

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SidComposite(Node):
    name: Str  # symbol
    location: NeighborChain
    entries: Set[SidCompositeEntry]

    # node private symbol table to entries
    @property
    def symbol_tbl(self):
        return {e.name: e for e in self.entries}

    @property
    def ptr_name(self):
        return self.name + "_ptrs"

    @property
    def origin_name(self):
        return self.name + "_origins"

    @property
    def strides_name(self):
        return self.name + "_strides"


class NeighborLoop(Stmt):
    body_location_type: common.LocationType
    body: List[Stmt]
    connectivity: Str  # symbol ref to Connectivity
    outer_sid: Str  # symbol ref to SidComposite where the neighbor tables lives (and sparse fields)
    sid: Optional[
        Str
    ]  # symbol ref to SidComposite where the fields of the loop body live (None if only sparse fields are accessed)


class Connectivity(Node):
    name: Str  # symbol name
    chain: NeighborChain


class Kernel(Node):
    # location_type: common.LocationType
    name: Str  # symbol decl table
    connectivities: List[Connectivity]
    sids: List[SidComposite]

    primary_connectivity: Str  # symbol ref to the above
    primary_sid: Str  # symbol ref to the above
    ast: List[Stmt]


class KernelCall(Node):
    name: Str  # symbol ref


class VerticalDimension(Node):
    pass


class UField(Node):
    name: Str
    vtype: common.DataType
    dimensions: List[Union[common.LocationType, NeighborChain, VerticalDimension]]  # Set?


class Temporary(UField):
    pass
    # name: Str
    # dimensions: List[Union[common.LocationType, NeighborChain, VerticalDimension]]  # Set?


class Computation(Node):
    name: Str
    parameters: List[UField]
    temporaries: List[Temporary]
    ctrlflow_ast: List[KernelCall]
    kernels: List[Kernel]  # probably replace by ctrlflow ast (where Kernel is one CtrlFlowStmt)
