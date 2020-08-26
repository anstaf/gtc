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
from typing import ForwardRef, List, Union

import gtc.common as common
from eve import Node


__all__ = [
    "GTScriptAstNode",
    "Statement",
    "Expr",
    "Symbol",
    "IterationOrder",
    "Constant",
    "Interval",
    "LocationSpecification",
    "SubscriptSingle",
    "SubscriptMultiple",
    "BinaryOp",
    "Call",
    "LocationComprehension",
    "Generator",
    "Assign",
    "Stencil",
    "Pass",
    "Argument",
    "Computation",
]


class GTScriptAstNode(Node):
    pass


class Statement(GTScriptAstNode):
    pass


class Expr(GTScriptAstNode):
    pass


class Symbol(Expr):
    name: str


class IterationOrder(GTScriptAstNode):
    order: str


# todo: use type parameter see https://github.com/samuelcolvin/pydantic/pull/595
# T = TypeVar('T')
# class Constant(GT4PyAstNode, Generic[T]):
#    value: T


class Constant(Expr):
    # todo: use StrictStr, StrictInt, StrictFloat as pydantic automatically converts into the first
    #  type it occurs. As a result currently all integers become floats
    value: Union[float, int, type(None), str]


class Interval(GTScriptAstNode):
    start: Constant  # todo: use Constant[Union[int, str, type(None)]]
    stop: Constant


# todo: allow interval(...) by introducing Optional(captures={...}) placeholder
# Optional(captures={start=0, end=None})


class LocationSpecification(GTScriptAstNode):
    name: Symbol
    location_type: str


# todo: proper cannonicalization (CanBeCanonicalizedTo[Subscript] ?)
class SubscriptSingle(Expr):
    value: Symbol
    index: str


SubscriptMultiple = ForwardRef("SubscriptMultiple")


class SubscriptMultiple(Expr):
    value: Symbol
    indices: List[Union[Symbol, SubscriptSingle, SubscriptMultiple]]


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class Call(Expr):
    args: List[Expr]
    func: str


# class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]


class LocationComprehension(GTScriptAstNode):
    target: Symbol
    iterator: Call


class Generator(Expr):
    generators: List[LocationComprehension]
    elt: Expr


class Assign(Statement):
    target: Union[Symbol, SubscriptSingle, SubscriptMultiple]
    value: Expr


Stencil = ForwardRef("Stencil")


class Stencil(GTScriptAstNode):
    iteration_spec: List[Union[IterationOrder, LocationSpecification, Interval]]
    body: List[Union[Statement, Stencil]]  # todo: stencil only allowed non-canonicalized


# class Attribute(GT4PyAstNode):
#    attr: str
#    value: Union[Attribute, Name]
#
#    @staticmethod
#    def template():
#        return ast.Attribute(attr=Capture("attr"), value=Capture("value"))


class Pass(Statement):
    pass


class Argument(GTScriptAstNode):
    name: str
    type_: Union[Symbol, Union[SubscriptMultiple, SubscriptSingle]]
    # is_keyword: bool


class Computation(GTScriptAstNode):
    name: str
    arguments: List[Argument]
    stencils: List[Stencil]
    # stencils: List[Union[Stencil[Stencil[Statement]], Stencil[Statement]]]
