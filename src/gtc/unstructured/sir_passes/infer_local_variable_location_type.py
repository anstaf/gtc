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

import eve  # noqa: F401
from eve import Node, NodeTranslator, NodeVisitor
from gtc import sir


class GTException(Exception):
    pass


class AnalysisException(GTException):
    def __init__(self, message=None, **kwargs):
        super().__init__(message)


class InferLocalVariableLocationTypeTransformation(NodeTranslator):
    """Visitor for updating LocationTypes from a dict of variable names to LocationTypes.

    The list of inferred LocationTypes is generated by the _LocationTypeAnalysis visitor.
    """

    @classmethod
    def apply(cls, root, **kwargs) -> Node:
        """Runs the  _LocationTypeAnalysis and updates VarDeclStmts with the result.

        Returns:
            New tree where all VarDeclStmts have a LocationType set
            or raises an exception if the LocationType couldn't be inferred.
        """
        inferred_location = _infer_location_type_of_local_variables(root)
        return cls().visit(root, inferred_location=inferred_location)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, *, inferred_location, **kwargs):
        if node.name not in inferred_location and node.location_type is None:
            raise AnalysisException("Cannot deduce location type for {}".format(node.name))
        node.location_type = inferred_location[node.name]
        return node


class _LocationTypeAnalysis(NodeVisitor):
    """Analyse local variable usage and infers location type if possible.

    Result of `infer_location_types()` is a dict of variable names to LocationType.

    Can deduce variable LocationTypes by assignments from:
    - Reductions (as they have a fixed location type)
    - Fields (as they have a fixed location type)
    - Variables recursively (by building a dependency tree)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.inferred_location = {}
        self.var_dependencies = {}  # depender -> set(dependees)
        self.sir_stencil_params = {}  # TODO symbol table: list of fields (stencil parameters)

    @classmethod
    def infer_location_types(cls, root, **kwargs) -> Node:
        """Runs the visitor to directly infer LocationTypes and propagates to dependent variables.

        The visitor pass infers LocationTypes from direct assignments and builds up a dependency graph of
        variable assignments. In postprocessing with `_propagate_location_type()` the directly inferred LocationType
        is propagated to other variables using the dependency graph.
        """
        instance = cls()
        instance.visit(root, **kwargs)
        original_inferred_location_keys = list(instance.inferred_location.keys())
        for var_name in original_inferred_location_keys:
            instance._propagate_location_type(var_name)
        return instance.inferred_location

    def _propagate_location_type(self, var_name: str):
        for dependee in self.var_dependencies[var_name]:
            if (
                dependee in self.inferred_location
                and self.inferred_location[dependee] != self.inferred_location[var_name]
            ):
                raise AnalysisException(
                    "Incompatible location type detected for {}".format(dependee)
                )
            else:
                self.inferred_location[dependee] = self.inferred_location[var_name]
                self._propagate_location_type(dependee)

    def _set_location_type(self, cur_var_name: str, location_type: sir.LocationType):
        if (
            cur_var_name in self.inferred_location
            and self.inferred_location[cur_var_name] != location_type
        ):
            raise RuntimeError("Incompatible location types deduced for {cur_var_name}")
        else:
            self.inferred_location[cur_var_name] = location_type

    def visit_Stencil(self, node: sir.Stencil, **kwargs):
        for f in node.params:
            self.sir_stencil_params[f.name] = f
        self.visit(node.ast)

    def visit_FieldAccessExpr(self, node: sir.FieldAccessExpr, **kwargs):
        if "cur_var" in kwargs:
            new_type = self.sir_stencil_params[
                node.name
            ].field_dimensions.horizontal_dimension.dense_location_type  # TODO use symbol table
            self._set_location_type(kwargs["cur_var"].name, new_type)

    def visit_VarAccessExpr(self, node: sir.VarAccessExpr, **kwargs):
        if "cur_var" in kwargs:
            # rhs of assignment/declaration
            if node.name not in self.var_dependencies:
                raise RuntimeError("{node.name} was not declared")
            self.var_dependencies[node.name].add(kwargs["cur_var"].name)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, **kwargs):
        if node.name in self.var_dependencies:
            raise RuntimeError("Redeclaration of variable")  # TODO symbol table will take care
        else:
            self.var_dependencies[node.name] = set()

        assert len(node.init_list) == 1
        self.visit(node.init_list[0], cur_var=node)

    def visit_ReductionOverNeighborExpr(self, node: sir.ReductionOverNeighborExpr, **kwargs):
        if "cur_var" in kwargs:
            self._set_location_type(kwargs["cur_var"].name, node.chain[0])

    def visit_AssignmentExpr(self, node: sir.AssignmentExpr, **kwargs):
        if isinstance(node.left, sir.VarAccessExpr):
            if "cur_var" in kwargs:
                raise RuntimeError(
                    "Variable assignment inside rhs of variable assignment is not supported."
                )
            else:
                self.visit(node.right, cur_var=node.left)


def _infer_location_type_of_local_variables(root: Node):
    return _LocationTypeAnalysis().infer_location_types(root)
