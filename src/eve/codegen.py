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

"""Tools for code generation."""


import abc
import collections.abc
import contextlib
import os
import re
import string
import sys
import textwrap
import types
from subprocess import PIPE, Popen

import black
import boltons.iterutils as bo_iterutils
import jinja2
from mako import template as mako_tpl

from . import typing, utils
from .concepts import Node
from .typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from .visitors import AnyTreeNode, NodeVisitor


try:
    import clang_format

    _CLANG_FORMAT_AVAILABLE = True
    del clang_format
except ImportError:
    _CLANG_FORMAT_AVAILABLE = False


SourceFormatter = Callable[[str], str]

SOURCE_FORMATTERS: Dict[str, SourceFormatter] = {}


def register_formatter(language: str,) -> Callable[[SourceFormatter], SourceFormatter]:
    def _decorator(formatter: SourceFormatter) -> SourceFormatter:
        if language in SOURCE_FORMATTERS:
            raise ValueError(f"Another formatter for language '{language}' already exists")

        assert callable(formatter)
        SOURCE_FORMATTERS[language] = formatter

        return formatter

    return _decorator


@register_formatter("python")
def format_python_source(
    source: str,
    *,
    line_length: int = 100,
    target_versions: Optional[Set[str]] = None,
    string_normalization: bool = True,
) -> str:
    target_versions = target_versions or f"{sys.version_info.major}{sys.version_info.minor}"
    target_versions = set(black.TargetVersion[f"PY{v.replace('.', '')}"] for v in target_versions)

    formatted_source = black.format_str(
        source,
        mode=black.FileMode(
            line_length=line_length,
            target_versions=target_versions,
            string_normalization=string_normalization,
        ),
    )
    assert isinstance(formatted_source, str)

    return formatted_source


if _CLANG_FORMAT_AVAILABLE:

    @register_formatter("cpp")
    def format_cpp_source(
        source: str,
        *,
        style: Optional[str] = None,
        fallback_style: Optional[str] = None,
        sort_includes: bool = False,
    ) -> str:
        args = ["clang-format"]
        if style:
            args.append(f"--style={style}")
        if fallback_style:
            args.append(f"--fallback-style={style}")
        if sort_includes:
            args.append("--sort-includes")

        p = Popen(args, stdout=PIPE, stdin=PIPE, encoding="utf8")
        formatted_source, _ = p.communicate(input=source)
        assert isinstance(formatted_source, str)

        return formatted_source


def format_source(language: str, source: str, *, skip_errors: bool = True, **kwargs: Any) -> str:
    formatter = SOURCE_FORMATTERS.get(language, None)
    try:
        if formatter:
            return formatter(source, **kwargs)  # type: ignore # Callable does not support **kwargs
        else:
            raise RuntimeError(f"Missing formatter for '{language}' language")
    except Exception as e:
        if skip_errors:
            return source
        else:
            raise RuntimeError(
                f"Something went wrong when trying to format '{language}' source code"
            ) from e


class Name:
    """Text string representing a symbol name in a programming language."""

    words: List[str]

    @classmethod
    def from_string(cls, name: str, case_style: utils.CaseStyleConverter.CASE_STYLE) -> "Name":
        return cls(utils.CaseStyleConverter.split(name, case_style))

    def __init__(self, words: utils.AnyWordsIterable) -> None:
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        words = [*words]
        if not all(isinstance(item, str) for item in words):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        self.words = words

    def as_case(self, case_style: utils.CaseStyleConverter.CASE_STYLE) -> str:
        return utils.CaseStyleConverter.join(self.words, case_style)


AnyTextSequence = Union[Sequence[str], "TextBlock"]


class TextBlock:
    """A block of source code represented as a sequence of text lines.

    This class also contains a context manager method (:meth:`indented`)
    for simple `indent - append - dedent` workflows.

    Args:
        indent_level: Initial indentation level
        indent_size: Number of characters per indentation level
        indent_char: Character used in the indentation
        end_line: Character or string used as new-line separator

    """

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ) -> None:
        if not isinstance(indent_char, str) or len(indent_char) != 1:
            raise ValueError("'indent_char' must be a single-character string")
        if not isinstance(end_line, str):
            raise ValueError("'end_line' must be a string")

        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines: List[str] = []

    def append(self, new_line: str, *, update_indent: int = 0) -> "TextBlock":
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        self.lines.append(self.indent_str + new_line)

        return self

    def extend(self, new_lines: AnyTextSequence, *, dedent: bool = False) -> "TextBlock":
        assert isinstance(new_lines, (collections.abc.Sequence, TextBlock))

        if dedent:
            if isinstance(new_lines, TextBlock):
                new_lines = textwrap.dedent(new_lines.text).splitlines()
            else:
                new_lines = textwrap.dedent("\n".join(new_lines)).splitlines()

        elif isinstance(new_lines, TextBlock):
            new_lines = new_lines.lines

        for line in new_lines:
            self.append(line)

        return self

    def empty_line(self, count: int = 1) -> "TextBlock":
        self.lines.extend([""] * count)
        return self

    def indent(self, steps: int = 1) -> "TextBlock":
        self.indent_level += steps
        return self

    def dedent(self, steps: int = 1) -> "TextBlock":
        assert self.indent_level >= steps
        self.indent_level -= steps
        return self

    @contextlib.contextmanager
    def indented(self, steps: int = 1) -> Iterator["TextBlock"]:
        self.indent(steps)
        yield self
        self.dedent(steps)

    @property
    def text(self) -> str:
        """Single string with the whole block contents."""
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    @property
    def indent_str(self) -> str:
        """Indentation string for new lines (in the current state)."""
        return self.indent_char * (self.indent_level * self.indent_size)

    def __iadd__(self, source_line: str) -> "TextBlock":
        return self.append(source_line)

    def __len__(self) -> int:
        return len(self.lines)

    def __str__(self) -> str:
        return self.text


def _compile_str_format_re() -> re.Pattern:
    prefix = r"((?:[^\^]|\^\^)*)"
    fmt_spec = "(.*)"

    joiner = "((?:[^:]|::)*)"
    item_fmt_spec = r"((?:[^\]]|\]\])*)"
    sequence = f"{joiner}(?::{item_fmt_spec})?"

    spec = rf"(?:{prefix}\^)?(?:\[{sequence}\])?:{fmt_spec}"

    return re.compile(spec)


class StringFormatter(string.Formatter):
    """StringFormatter

    s = "prefix^[joiner:item_fmt]:fmt"
    """

    __FORMAT_RE = _compile_str_format_re()

    def format_field(self, value: Any, format_spec: str) -> Any:
        m = self.__FORMAT_RE.match(format_spec)
        if not m:
            return super().format_field(value, format_spec)

        prefix = m[1]
        joiner = m[2]
        item_format_spec = m[3]
        format_spec = m[4]

        if joiner is not None:
            joiner = joiner.replace("::", ":")
            if not bo_iterutils.is_collection(value):
                raise ValueError(f"Collection formatting used with a scalar value {value}")

            sequence = value
            item_format_spec = item_format_spec.replace("]]", "]") if item_format_spec else ""
            formatted = joiner.join(
                super().format_field(item, item_format_spec) for item in sequence
            )
        else:
            formatted = value

        if format_spec:
            formatted = super().format_field(formatted, format_spec)

        if prefix:
            prefix = prefix.replace("^^", "^")
            formatted = textwrap.indent(formatted, prefix)

        return formatted


TemplateT = TypeVar("TemplateT", bound="Template")


@typing.runtime_checkable
class Template(Protocol):
    """Master Template class (to be subclasssed).

    Subclassess must implement the `__init__` (with a type annotation for `definition`)
    and `_render` methods.

    """

    @classmethod
    def from_file(cls: Type[TemplateT], file_path: Union[str, os.PathLike]) -> "Template":
        if cls is Template:
            raise RuntimeError("This method can only be called in concrete Template subclasses")

        with open(file_path, "r") as f:
            definition = f.read()
        return cls(definition)

    def render(self, mapping: Optional[Mapping[str, str]] = None, **kwargs: Any) -> str:
        """Render the template.

        Args:
            mapping (optional): A `dict` whose keys match the template placeholders.
            **kwargs: placeholder values might be also provided as
                keyword arguments, and they should take precedence over ``mapping``
                values for duplicated keys.

        """
        if not mapping:
            mapping = {}
        if kwargs:
            mapping = {**mapping, **kwargs}

        return self.render_template(**mapping)

    @abc.abstractmethod
    def __init__(self, definition: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def render_template(self, **kwargs: Any) -> str:
        pass


class StrFormatTemplate(Template):
    definition: str

    _formatter_: ClassVar[StringFormatter] = StringFormatter()

    def __init__(self, definition: str, **kwargs: Any) -> None:
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self._formatter_.format(self.definition, **kwargs)


class StringTemplate(Template):
    definition: string.Template

    def __init__(self, definition: string.Template, **kwargs: Any) -> None:
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self.definition.substitute(**kwargs)


class JinjaTemplate(Template):
    definition: jinja2.Template

    def __init__(self, definition: jinja2.Template, **kwargs: Any) -> None:
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self.definition.render(**kwargs)


class MakoTemplate(Template):
    definition: mako_tpl.Template

    def __init__(self, definition: mako_tpl.Template, **kwargs: Any) -> None:
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        result = self.definition.render(**kwargs)
        assert isinstance(result, str)
        return result


class TemplatedGenerator(NodeVisitor):
    """A code generator visitor using :class:`TextTemplate` s."""

    _templates_: ClassVar[Mapping[str, Template]]

    @classmethod
    def __init_subclass__(cls, *, inherit_templates: bool = True, **kwargs: Any) -> None:
        # mypy has troubles with __init_subclass__: https://github.com/python/mypy/issues/4660
        super().__init_subclass__(**kwargs)  # type: ignore
        if "_templates_" in cls.__dict__:
            raise TypeError(f"Invalid '_templates_' member in class {cls}")

        templates: Dict[str, Template] = {}
        if inherit_templates:
            for templated_gen_class in reversed(cls.__mro__):
                if isinstance(templated_gen_class, TemplatedGenerator):
                    templates.update(templated_gen_class._templates_)

        templates.update(
            {
                key: value
                for key, value in cls.__dict__.items()
                if isinstance(value, Template) and not key.startswith("_")
            }
        )

        cls._templates_ = types.MappingProxyType(templates)

    @classmethod
    def apply(cls, root: AnyTreeNode, **kwargs: Any) -> Union[str, Collection[str]]:
        """Public method to build a class instance and visit an IR node.

        The order followed to choose a `dump()` function for instances of
        :class:`eve.Node` is the following:

            1. A `self.visit_NODE_TYPE_NAME()` method where `NODE_TYPE_NAME`
               matches `NODE_CLASS.__name__`, and `NODE_CLASS` is the
               actual type of the node or any of its superclasses
               following MRO order.
            2. A `Templates.NODE_TYPE_NAME` template where `NODE_TYPE_NAME`
               matches `NODE_CLASS.__name__`, and `NODE_CLASS` is the
               actual type of the node or any of its superclasses
               following MRO order.

        When a template is used, the following keys will be passed to the template
        instance:

            * `**node_fields`: all the node children and attributes by name.
            * `_attrs`: a `dict` instance with the results of visiting all
              the node attributes.
            * `_children`: a `dict` instance with the results of visiting all
              the node children.
            * `_this_node`: the actual node instance (before visiting children).
            * `_this_generator`: the current generator instance.
            * `_this_module`: the generator's module instance .
            * `**kwargs`: the keyword arguments received by the visiting method.

        For primitive types (not :class:`eve.Node` subclasses),
        the :meth:`self.generic_dump()` method will be used.

        Args:
            root: An IR node.
            node_templates (optiona): see :class:`NodeDumper`.
            dump_function (optiona): see :class:`NodeDumper`.
            **kwargs (optional): custom extra parameters forwarded to
                `visit_NODE_TYPE_NAME()`.

        Returns:
            String (or collection of strings) with the dumped version of the root IR node.

        """
        return typing.cast(Union[str, Collection[str]], cls().visit(root, **kwargs))

    @classmethod
    def generic_dump(cls, node: AnyTreeNode, **kwargs: Any) -> str:
        """Class-specific ``dump()`` function for primitive types.

        This class could be redefined in the subclasses.
        """
        return str(node)

    def generic_visit(self, node: AnyTreeNode, **kwargs: Any) -> Union[str, Collection[str]]:
        result: Union[str, Collection[str]] = ""
        if isinstance(node, Node):
            template, _ = self.get_template(node)
            if template:
                result = self.render_template(
                    template,
                    node,
                    self.transform_children(node, **kwargs),
                    self.transform_attrs(node, **kwargs),
                    **kwargs,
                )
        elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)) and not isinstance(
            node, self.ATOMIC_COLLECTION_TYPES
        ):
            result = [self.visit(value, **kwargs) for value in node]
        elif isinstance(node, collections.abc.Mapping):
            result = {key: self.visit(value, **kwargs) for key, value in node.items()}
        else:
            result = self.generic_dump(node, **kwargs)

        return result

    def get_template(self, node: AnyTreeNode) -> Tuple[Optional[Template], Optional[str]]:
        """Get a template for a node instance (see :meth:`apply`)."""
        template: Optional[Template] = None
        template_key: Optional[str] = None
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                template_key = node_class.__name__
                template = self._templates_.get(template_key, None)
                if template is not None or node_class is Node:
                    break

        return template, None if template is None else template_key

    def render_template(
        self,
        template: Template,
        node: Node,
        transformed_children: Mapping[str, Any],
        transformed_attrs: Mapping[str, Any],
        **kwargs: Any,
    ) -> str:
        """Render a template using node instance data (see :meth:`apply`)."""

        return template.render(
            **transformed_children,
            **transformed_attrs,
            _children=transformed_children,
            _attrs=transformed_attrs,
            _this_node=node,
            _this_generator=self,
            _this_module=sys.modules[type(self).__module__],
            **kwargs,
        )

    def transform_children(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.iter_children()}

    def transform_attrs(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.iter_attributes()}
