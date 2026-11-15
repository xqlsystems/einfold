# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Compact arrays for [DataFusion](https://datafusion.apache.org/) SQL plans.

Coordinates describe axes once. Numerical expressions retain their axes
and share values until a relational output is requested. Unsupported row
semantics raise ``_Decline`` before any public parameter is changed.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

try:
    from scipy.special import erf as _erf
except ImportError:
    _erf = np.vectorize(math.erf, otypes=[np.float64])

_EXECUTION = ContextVar("einfold_dense_execution", default=None)
_MAX_BYTES = ContextVar("einfold_dense_array_limit", default=1 << 30)
_CACHE = ContextVar("einfold_dense_computed_cache", default=None)


class _ComputedCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.peak = 0
        self.evictions = 0
        self.entries = OrderedDict()

    def remember(self, tensor):
        value = tensor.cached
        if tensor.kind == "array":
            return value
        size = value.nbytes
        while self.entries and self.bytes + size > self.max_bytes:
            old, amount = self.entries.popitem(last=False)
            old.cached = None
            self.bytes -= amount
            self.evictions += 1
        if size <= self.max_bytes:
            self.entries[tensor] = size
            self.bytes += size
            self.peak = max(self.peak, self.bytes)
        else:
            tensor.cached = None
        return value


def _remember(tensor):
    cache = _CACHE.get()
    return cache.remember(tensor) if cache is not None else tensor.cached


def _numpy(value):
    runtime = _EXECUTION.get()
    return runtime.numpy(value) if runtime is not None else np.asarray(value)


class _Decline(Exception):
    """This region requires the original relational implementation."""


@dataclass(eq=False, frozen=True)
class _Axis:
    values: np.ndarray

    def __len__(self):
        return len(self.values)


def _axes(*groups):
    return tuple(dict.fromkeys(itertools.chain.from_iterable(groups)))


def _align(array, source, target):
    ordered = tuple(a for a in target if a in source)
    if set(source) - set(target):
        raise _Decline("an expression depends on an unavailable axis")
    permutation = tuple(source.index(a) for a in ordered)
    if permutation != tuple(range(len(source))):
        array = array.transpose(permutation)
    return array.reshape(tuple(len(a) if a in source else 1 for a in target))


class _Tensor:
    def __init__(self, kind, axes=(), args=(), value=None, axis=None):
        self.kind, self.axes, self.args = kind, tuple(axes), tuple(args)
        self.value, self.axis = value, axis
        self.cached = None
        self.domain = axis
        self.count_bound = None
        self.host_only = (
            np.asarray(value).dtype.kind in "biuUS"
            if kind == "array"
            else kind not in {"reduce", "device"}
            and all(a.host_only for a in args)
        )
        if kind in {"rename", "take", "int64"} and args:
            self.domain = args[0].domain
            self.count_bound = args[0].count_bound
        if kind == "where":
            domains = [a.domain for a in args[1:] if a.domain is not None]
            if (
                domains
                and all(
                    np.array_equal(d.values, domains[0].values) for d in domains
                )
                and all(
                    a.domain is not None
                    or (
                        a.kind == "array"
                        and not a.axes
                        and a.value in domains[0].values
                    )
                    for a in args[1:]
                )
            ):
                self.domain = domains[0]

    def get(self):
        if self.kind == "device":
            return self.value
        if self.kind == "rename":
            return self.args[0].get()
        cache = _CACHE.get()
        if (
            self.cached is not None
            and cache is not None
            and self in cache.entries
        ):
            cache.entries.move_to_end(self)
        if self.cached is None:
            if (
                self.kind not in {"array", "rename"}
                and math.prod(len(a) for a in self.axes) * 8 > _MAX_BYTES.get()
            ):
                raise _Decline(
                    f"dense temporary exceeds array limit: {self.kind}, shape={tuple(len(a) for a in self.axes)}"
                )
            if self.kind == "array":
                runtime = _EXECUTION.get()
                self.cached = (
                    runtime.array(self.value)
                    if runtime is not None
                    else np.asarray(self.value)
                )
            elif self.kind == "take":
                source, indexes, at, replacement = self.value
                parent = self.args[0]
                overlap = set(replacement) & (set(parent.axes) - {source})
                if overlap:
                    selectors = []
                    for axis in parent.axes:
                        if axis is source:
                            selectors.append(
                                _align(indexes, replacement, self.axes)
                            )
                        else:
                            selectors.append(
                                _align(np.arange(len(axis)), (axis,), self.axes)
                            )
                    runtime = _EXECUTION.get()
                    self.cached = (
                        runtime.index(parent.get(), selectors)
                        if runtime is not None
                        else parent.get()[tuple(selectors)]
                    )
                else:
                    runtime = _EXECUTION.get()
                    self.cached = (
                        runtime.take(parent.get(), indexes, at)
                        if runtime is not None
                        else np.take(parent.get(), indexes, axis=at)
                    )
            elif self.kind == "reduce":
                self.cached = self.value()
            else:
                runtime = _EXECUTION.get()
                if runtime is not None:
                    result = runtime.pointwise(self)
                    if result is not None:
                        self.cached = result
                        return _remember(self)
                arrays = [_align(a.get(), a.axes, self.axes) for a in self.args]
                with np.errstate(all="ignore"):
                    self.cached = _OPS[self.kind](*arrays)
            return _remember(self)
        return self.cached

    def rename(self, mapping):
        if not any(a in mapping for a in self.axes):
            return self
        return _Tensor(
            "rename",
            tuple(mapping.get(a, a) for a in self.axes),
            (self,),
            axis=mapping.get(self.axis, self.axis),
        )

    def take(self, axis, indexes, replacement):
        if axis not in self.axes:
            return self
        at = self.axes.index(axis)
        new_axes = _axes(self.axes[:at], replacement, self.axes[at + 1 :])
        result = _Tensor(
            "take", new_axes, (self,), (axis, indexes, at, replacement)
        )
        runtime = _EXECUTION.get()
        if runtime is not None and runtime.owns(indexes):
            result.host_only = False
        return result


def _const(value):
    return _Tensor("array", value=value)


def _truth(value):
    if value.host_only:
        flags = value.get()
        if flags.all():
            return _const(True)
        if not flags.any():
            return _const(False)
    return value


def _mask(values, axes):
    runtime = _EXECUTION.get()
    if runtime is not None and runtime.owns(values):
        return _Tensor("device", axes, value=values)
    return _truth(_Tensor("array", axes, value=values))


def _is(value, literal):
    return value.kind == "array" and not value.axes and value.value == literal


def _op(kind, *args):
    if kind in {"+", "-", "*", "/", "%", "neg", "abs", "power", "pow"} and any(
        a.domain is not None for a in args
    ):
        runtime = _EXECUTION.get()
        if runtime is not None and any(
            a.domain is not None and runtime.owns(a.get()) for a in args
        ):
            raise _Decline(
                "arithmetic on device index columns requires an explicit floating cast"
            )
    if (
        kind == ">"
        and args[0].count_bound is not None
        and args[1].kind == "array"
        and not args[1].axes
        and np.asarray(args[1].value).dtype.kind in "biuf"
    ) and args[0].count_bound <= args[1].value:
        return _const(False)
    if kind == "and":
        if any(_is(a, False) for a in args):
            return _const(False)
        args = tuple(a for a in args if not _is(a, True))
        if not args:
            return _const(True)
        if len(args) == 1:
            return args[0]
    if kind == "or":
        if any(_is(a, True) for a in args):
            return _const(True)
        args = tuple(a for a in args if not _is(a, False))
        if not args:
            return _const(False)
        if len(args) == 1:
            return args[0]
    if kind in {"and", "or"} and len(args) > 2:
        return _op(kind, args[0], _op(kind, *args[1:]))
    if kind == "where":
        if args[1] is args[2]:
            return args[1]
        if (
            all(
                a.kind == "array"
                and not a.axes
                and np.asarray(a.value).dtype.kind == "b"
                for a in args[1:]
            )
            and args[1].value == args[2].value
        ):
            return args[1]
        if _is(args[0], True):
            return args[1]
        if _is(args[0], False):
            return args[2]
    return _Tensor(kind, _axes(*(a.axes for a in args)), args)


def _divide(a, b):
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        if np.any(b == 0) or np.any((a == np.iinfo(np.int64).min) & (b == -1)):
            raise _Decline("exceptional integer division")
        quotient, remainder = np.divmod(a, b)
        return quotient + ((remainder != 0) & ((a < 0) != (b < 0)))
    return np.divide(a, b)


def _remainder(a, b):
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        return a - _divide(a, b) * b
    return np.fmod(a, b)


def _arithmetic(kind, a, b):
    operation = {"+": np.add, "-": np.subtract, "*": np.multiply}[kind]
    if a.dtype.kind in "iu" and b.dtype.kind in "iu" and a.size and b.size:
        if a.dtype.kind == "u" or b.dtype.kind == "u":
            raise _Decline("unsigned integer arithmetic")
        aa = [int(a.min()), int(a.max())]
        bb = [int(b.min()), int(b.max())]
        bounds = [
            {"+": x + y, "-": x - y, "*": x * y}[kind] for x in aa for y in bb
        ]
        limits = np.iinfo(np.result_type(a.dtype, b.dtype))
        if min(bounds) < limits.min or max(bounds) > limits.max:
            raise _Decline("potential integer arithmetic overflow")
    return operation(a, b)


def _negative(value):
    if value.dtype.kind == "u" or (
        value.dtype.kind == "i" and np.any(value == np.iinfo(value.dtype).min)
    ):
        raise _Decline("exceptional integer negation")
    return np.negative(value)


def _int64(value):
    if value.dtype.kind not in "biuf":
        raise _Decline("integer cast requires SQL conversion")
    if value.dtype.kind == "f" and (
        not np.isfinite(value).all()
        or np.any(value >= 1 << 63)
        or np.any(value < -(1 << 63))
    ):
        raise _Decline("floating-to-integer cast outside its exact range")
    if value.dtype.kind == "u" and np.any(value >= 1 << 63):
        raise _Decline("unsigned cast exceeds int64")
    return value.astype(np.int64)


def _abs(value):
    if value.dtype.kind == "i" and np.any(value == np.iinfo(value.dtype).min):
        raise _Decline("exceptional integer absolute value")
    return np.abs(value)


def _power(a, b):
    return np.power(
        a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    )


_OPS = {
    "+": lambda a, b: _arithmetic("+", a, b),
    "-": lambda a, b: _arithmetic("-", a, b),
    "*": lambda a, b: _arithmetic("*", a, b),
    "/": _divide,
    "%": _remainder,
    "=": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "and": np.logical_and,
    "or": np.logical_or,
    "not": np.logical_not,
    "neg": _negative,
    "where": np.where,
    "abs": _abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "erf": _erf,
    "ln": np.log,
    "log": np.log10,
    "tanh": np.tanh,
    "isnan": np.isnan,
    "iszero": lambda x: x == 0,
    "signum": np.sign,
    "sign": np.sign,
    "sin": np.sin,
    "cos": np.cos,
    "power": _power,
    "pow": _power,
    "floor": np.floor,
    "ceil": np.ceil,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "float64": lambda x: x.astype(np.float64),
    "int64": _int64,
    "bool": lambda x: x.astype(bool),
}


@dataclass
class _Column:
    tensor: _Tensor
    valid: _Tensor

    def rename(self, mapping):
        return _Column(self.tensor.rename(mapping), self.valid.rename(mapping))


def _column(value, axes=(), axis=None):
    return _Column(_Tensor("array", axes, value=value, axis=axis), _const(True))


class _Table:
    def __init__(self, axes=(), columns=None, present=None):
        self.axes = tuple(axes)
        self.columns = OrderedDict(columns or [])
        self.present = present if present is not None else _const(True)

    def resolve(self, relation, name):
        if (relation, name) in self.columns:
            return self.columns[relation, name]
        if relation is None:
            matches = [c for (_, n), c in self.columns.items() if n == name]
            if len(matches) == 1:
                return matches[0]
        raise _Decline(f"unbound dense column {relation}.{name}")

    def alias(self, relation, names=None):
        if names is None:
            names = [n for _, n in self.columns]
        else:
            prefix = relation + "."
            names = [n.removeprefix(prefix) for n in names]
        if len(names) != len(self.columns) or len(set(names)) != len(names):
            raise _Decline("duplicate or changed alias fields")
        return _Table(
            self.axes,
            [((relation, n), v) for n, v in zip(names, self.columns.values())],
            self.present,
        )

    def rename(self, mapping):
        return _Table(
            tuple(mapping.get(a, a) for a in self.axes),
            [(k, v.rename(mapping)) for k, v in self.columns.items()],
            self.present.rename(mapping),
        )

    def take(self, axis, indexes, replacement):
        new_axes = _axes(
            *(replacement if a is axis else (a,) for a in self.axes)
        )
        columns = []
        for key, col in self.columns.items():
            value = col.tensor.take(axis, indexes, replacement)
            if col.tensor.axis is axis and len(replacement) == 1:
                new = replacement[0]
                runtime = _EXECUTION.get()
                if not (
                    runtime is not None and runtime.owns(indexes)
                ) and np.array_equal(axis.values[indexes], new.values):
                    value = _Tensor("array", (new,), value=new.values, axis=new)
            columns.append(
                (
                    key,
                    _Column(value, col.valid.take(axis, indexes, replacement)),
                )
            )
        return _Table(
            new_axes, columns, self.present.take(axis, indexes, replacement)
        )

    def filter(self, predicate):
        value = predicate.tensor
        # Select constant coordinate ranges before contractions. A single
        # layer selection stays a one-element axis, never twelve matrices
        # multiplied by a zero/one layer mask.
        if value.kind == "=" and _is(predicate.valid, True):
            for key, literal in (value.args, value.args[::-1]):
                if key.axis is not None and not literal.axes:
                    indexes = np.flatnonzero(key.axis.values == literal.get())
                    axis = _Axis(key.axis.values[indexes])
                    return self.take(key.axis, indexes, (axis,))
        return _Table(
            self.axes,
            self.columns.items(),
            _op("and", self.present, predicate.valid, value),
        )

    def arrow(self, schema):
        shape = tuple(len(a) for a in self.axes)
        present = np.broadcast_to(
            _align(_numpy(self.present.get()), self.present.axes, self.axes),
            shape,
        ).ravel()
        all_present = bool(present.all())
        if not present.any():
            return pa.Table.from_batches([], schema=schema)
        arrays = []
        if len(schema) != len(self.columns):
            raise _Decline("dense result schema changed")
        for field, col in zip(schema, self.columns.values()):
            values = np.broadcast_to(
                _align(_numpy(col.tensor.get()), col.tensor.axes, self.axes),
                shape,
            ).ravel()
            valid = np.broadcast_to(
                _align(_numpy(col.valid.get()), col.valid.axes, self.axes),
                shape,
            ).ravel()
            if not all_present:
                values, valid = values[present], valid[present]
            if (
                pa.types.is_floating(field.type)
                and not np.isfinite(values[valid]).all()
            ):
                raise _Decline(
                    "exceptional dense output requires SQL arithmetic"
                )
            arrays.append(pa.array(values, type=field.type, mask=~valid))
        return pa.Table.from_arrays(arrays, schema=schema)


def _read_dataset(name, dataset):
    axes = {}
    columns = []
    for dim, size in dataset.sizes.items():
        values = (
            np.asarray(dataset.coords[dim].values)
            if dim in dataset.coords
            else np.arange(size)
        )
        if (
            values.ndim != 1
            or values.dtype.kind not in "iuUS"
            or len(np.unique(values)) != size
        ):
            raise _Decline("coordinate domain is not a unique supported axis")
        axis = axes[dim] = _Axis(values)
        columns.append(((name, dim), _column(values, (axis,), axis)))
    for var, value in dataset.data_vars.items():
        data = value.data
        if not isinstance(data, np.ndarray) or data.dtype.kind not in "biuf":
            raise _Decline("source requires a NumPy numerical buffer")
        if data.dtype.kind == "f" and (
            data.dtype != np.float64 or not np.isfinite(data).all()
        ):
            raise _Decline("source requires finite float64 values")
        columns.append(
            ((name, var), _column(data, tuple(axes[d] for d in value.dims)))
        )
    return _Table(tuple(axes.values()), columns)


def _factors(value):
    if value.kind == "rename":
        parent = value.args[0]
        mapping = dict(zip(parent.axes, value.axes))
        return [factor.rename(mapping) for factor in _factors(parent)]
    if value.kind == "*":
        return [factor for child in value.args for factor in _factors(child)]
    if value.kind == "where" and _is(value.args[2], 0):
        # Keep a conditional product as a contraction with a mask factor.
        # Otherwise attention's backward pass expands B*H*T*T*C values.
        # Exceptional factors/outputs retain the normal SQL fallback.
        return _factors(value.args[1]) + [value.args[0]]
    if value.kind == "where" and _is(value.args[1], 0):
        return _factors(value.args[2]) + [_op("not", value.args[0])]
    return [value]


def _sum(value, mask, row_axes, kept):
    factors = _factors(value)
    if not _is(mask, True):
        factors.append(mask)
    used = _axes(*(v.axes for v in factors))
    output = tuple(a for a in kept if a in used)
    repeated = math.prod(
        len(a) for a in row_axes if a not in used and a not in kept
    )
    labels = {a: i for i, a in enumerate(used)}
    if len(labels) > 50:
        raise _Decline("too many contraction axes")

    def compute():
        runtime = _EXECUTION.get()
        if runtime is not None:
            arrays = [factor.get() for factor in factors]
            if any(runtime.owns(a) for a in arrays):
                return (
                    runtime.contract(
                        arrays,
                        [
                            [labels[a] for a in factor.axes]
                            for factor in factors
                        ],
                        [labels[a] for a in output],
                    )
                    * repeated
                )
        operands = []
        integer_bound = 1
        all_integer = True
        if len(_factors(value)) > 1:
            data_types = [f.get().dtype for f in _factors(value)]
            if all(d.kind in "biu" for d in data_types) and any(
                d.kind == "i" and d.itemsize < 8 for d in data_types
            ):
                raise _Decline(
                    "narrow integer product requires SQL overflow checks"
                )
        for factor in factors:
            data = factor.get()
            if data.dtype == np.bool_:
                data = data.astype(np.int64)
            if data.dtype.kind == "u":
                raise _Decline("unsigned integer contraction")
            if data.dtype.kind == "i" and data.size:
                integer_bound *= max(
                    1, abs(int(data.min())), abs(int(data.max()))
                )
                data = data.astype(np.int64, copy=False)
            else:
                all_integer = False
            if (
                np.issubdtype(data.dtype, np.floating)
                and not np.isfinite(data).all()
            ):
                raise _Decline("exceptional contraction factor")
            operands.extend([data, [labels[a] for a in factor.axes]])
        if (
            all_integer
            and integer_bound
            * math.prod(len(a) for a in row_axes if a not in kept)
            >= 1 << 63
        ):
            raise _Decline("potential integer aggregate overflow")
        operands.append([labels[a] for a in output])
        return np.einsum(*operands, optimize=True) * repeated

    result = _Tensor("reduce", output, value=compute)
    result.host_only = value.host_only and mask.host_only
    if _is(value, 1):
        result.count_bound = math.prod(
            len(a) for a in row_axes if a not in kept
        )
    return result


def _reduce(kind, column, table, kept):
    mask = _op("and", table.present, column.valid)
    value_input = column.tensor
    if kind == "avg" and getattr(value_input.get().dtype, "kind", "f") in "biu":
        value_input = _op("float64", value_input)
    summed = _sum(value_input, mask, table.axes, kept)
    count = _sum(_const(1), mask, table.axes, kept)
    if kind == "sum":
        value = summed
    elif kind == "avg":
        value = _op("/", summed, count)
    elif kind == "count":
        return _Column(count, _const(True))
    elif kind in ("max", "min"):
        used = _axes(column.tensor.axes, mask.axes)
        output = tuple(a for a in kept if a in used)

        def compute():
            runtime = _EXECUTION.get()
            if runtime is not None and runtime.owns(column.tensor.get()):
                return runtime.extreme(
                    kind,
                    column.tensor.get(),
                    column.tensor.axes,
                    mask.get(),
                    mask.axes,
                    used,
                    kept,
                )
            shape = tuple(len(a) for a in used)
            values = np.broadcast_to(
                _align(column.tensor.get(), column.tensor.axes, used), shape
            )
            valid = np.broadcast_to(_align(mask.get(), mask.axes, used), shape)
            if values.dtype.kind in "iu":
                limits = np.iinfo(values.dtype)
                initial = limits.min if kind == "max" else limits.max
            elif values.dtype.kind == "b":
                initial = kind != "max"
            else:
                initial = -np.inf if kind == "max" else np.inf
            values = np.where(valid, values, initial)
            axes = tuple(i for i, a in enumerate(used) if a not in kept)
            return (np.max if kind == "max" else np.min)(
                values, axis=axes, initial=initial
            )

        value = _Tensor("reduce", output, value=compute)
    else:
        raise _Decline(f"aggregate {kind}")
    return _Column(value, _truth(_op(">", count, _const(0))))


def _key(expr):
    kind = expr.variant_name()
    if kind == "Column":
        col = expr.to_variant()
        return col.relation(), col.name()
    return None, expr.schema_name()


def _expr(expr, table):
    kind = expr.variant_name()
    if kind == "Column":
        return table.resolve(*_key(expr))
    if kind == "Literal":
        value = expr.python_value()
        if isinstance(value, pa.Scalar):
            value = value.as_py()
        return _Column(
            _const(0 if value is None else value), _const(value is not None)
        )
    if kind == "Alias":
        return _expr(expr.to_variant().expr(), table)
    if kind in {"IsNull", "IsNotNull"}:
        col = _expr(expr.to_variant().expr(), table)
        return _Column(
            col.valid if kind == "IsNotNull" else _op("not", col.valid),
            _const(True),
        )
    if kind == "Cast":
        col = _expr(expr.to_variant().expr(), table)
        match = re.search(r" AS (Float64|Int64|Boolean)\)\)$", str(expr))
        if match is None:
            raise _Decline(f"unsupported cast {expr}")
        target = {"Float64": "float64", "Int64": "int64", "Boolean": "bool"}[
            match[1]
        ]
        if (
            target == "int64"
            and col.tensor.kind == "array"
            and not col.tensor.axes
        ):
            return _Column(
                _const(_int64(np.asarray(col.tensor.value))), col.valid
            )
        if (
            target == "int64"
            and _EXECUTION.get() is not None
            and not col.tensor.host_only
        ):
            raise _Decline(
                "GPU floating-to-integer cast requires SQL execution"
            )
        # Coordinate casts are accepted only when they leave the values exact.
        if col.tensor.axis is not None and target == "int64":
            values = col.tensor.axis.values
            if values.dtype.kind not in "iu" or (
                values.size
                and (
                    int(values.min()) < -(1 << 63)
                    or int(values.max()) >= 1 << 63
                )
            ):
                raise _Decline("coordinate cast changes its domain")
            return _Column(
                _Tensor(
                    "int64",
                    col.tensor.axes,
                    (col.tensor,),
                    axis=col.tensor.axis,
                ),
                col.valid,
            )
        return _Column(_op(target, col.tensor), col.valid)
    if kind == "Case":
        case = expr.to_variant()
        if case.expr() is not None:
            raise _Decline("simple CASE")
        result = (
            _expr(case.else_expr(), table)
            if case.else_expr() is not None
            else _Column(_const(0), _const(False))
        )
        for when, then in reversed(case.when_then_expr()):
            condition, value = _expr(when, table), _expr(then, table)
            yes = _op("and", condition.valid, condition.tensor)
            result = _Column(
                _op("where", yes, value.tensor, result.tensor),
                _op("where", yes, value.valid, result.valid),
            )
        return result
    if kind in {"Not", "Negative"}:
        col = _expr(expr.to_variant().expr(), table)
        return _Column(
            _op("not" if kind == "Not" else "neg", col.tensor), col.valid
        )
    if kind not in {"BinaryExpr", "ScalarFunction"}:
        raise _Decline(f"expression {kind}: {expr}")
    op = expr.rex_call_operator().lower()
    args = [_expr(a, table) for a in expr.rex_call_operands()]
    if op == "ddx_stop_gradient":
        return args[0]
    if op == "coalesce":
        result = args[-1]
        for col in reversed(args[:-1]):
            result = _Column(
                _op("where", col.valid, col.tensor, result.tensor),
                _op("or", col.valid, result.valid),
            )
        return result
    if op == "nullif":
        a, b = args
        equal = _op("and", b.valid, _op("=", a.tensor, b.tensor))
        return _Column(a.tensor, _op("and", a.valid, _op("not", equal)))
    if op in {"is not distinct from", "is distinct from"}:
        a, b = args
        both = _op("and", a.valid, b.valid)
        neither = _op("and", _op("not", a.valid), _op("not", b.valid))
        value = _op(
            "or", neither, _op("and", both, _op("=", a.tensor, b.tensor))
        )
        return _Column(
            value if op == "is not distinct from" else _op("not", value),
            _const(True),
        )
    if op in {"and", "or"}:
        a, b = args
        decisive = op != "and"
        valid = _op(
            "or",
            _op("and", a.valid, b.valid),
            _op(
                "or",
                _op("and", a.valid, _op("=", a.tensor, _const(decisive))),
                _op("and", b.valid, _op("=", b.tensor, _const(decisive))),
            ),
        )
    else:
        valid = _op("and", *(a.valid for a in args))
    if op not in _OPS:
        raise _Decline(f"function {op}")
    binary = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "and",
        "or",
        "pow",
        "power",
        "minimum",
        "maximum",
    }
    if len(args) != (2 if op in binary else 1):
        raise _Decline(f"unsupported arity for {op}")
    return _Column(_op(op, *(a.tensor for a in args)), valid)


def _conjuncts(expr):
    if (
        expr.variant_name() == "BinaryExpr"
        and expr.rex_call_operator().lower() == "and"
    ):
        return [p for arg in expr.rex_call_operands() for p in _conjuncts(arg)]
    return [expr]


def _indexes(axis, values):
    runtime = _EXECUTION.get()
    if runtime is not None and runtime.owns(values):
        return runtime.indexes(axis.values, values)
    order = np.argsort(axis.values)
    at = np.searchsorted(axis.values[order], values)
    if not len(order):
        raise _Decline("empty gather domain")
    indexes = order[np.minimum(at, len(order) - 1)]
    return indexes, axis.values[indexes] == values


def _key_indexes(axis, tensor):
    indexes, found = _indexes(axis, tensor.get())
    if (
        tensor.domain is not None
        and np.isin(tensor.domain.values, axis.values).all()
    ):
        found = np.ones(tuple(len(a) for a in tensor.axes), dtype=bool)
    return indexes, found


def _join(left, right, predicates, how):
    how = str(how)
    if how not in {"Inner", "Left"}:
        raise _Decline(f"join {how}")
    right = right.rename({a: _Axis(a.values) for a in right.axes})
    remaining = []
    for predicate in predicates:
        if isinstance(predicate, tuple):
            a, b = predicate
        elif (
            predicate.variant_name() != "BinaryExpr"
            or predicate.rex_call_operator().lower()
            not in {"=", "is not distinct from"}
        ):
            remaining.append(predicate)
            continue
        else:
            a, b = predicate.rex_call_operands()
        try:
            lc, rc = _expr(a, left), _expr(b, right)
        except _Decline:
            try:
                lc, rc = _expr(b, left), _expr(a, right)
            except _Decline:
                remaining.append(predicate)
                continue
        la, ra = lc.tensor.axis, rc.tensor.axis
        if la is not None or ra is not None:
            if not _is(lc.valid, True) and not _is(rc.valid, True):
                raise _Decline("join of two nullable coordinate keys")
            # A NULL coordinate cannot match a proven non-NULL key, under
            # either equality or null-safe equality. Keep unmatched left
            # rows for an outer join and mask their right-side columns.
            if how == "Inner":
                left.present = _op("and", left.present, lc.valid)
            right.present = _op(
                "and",
                right.present,
                rc.valid,
                lc.valid if how == "Left" else _const(True),
            )
        if la is not None and ra is not None:
            if np.array_equal(la.values, ra.values):
                right = right.rename({ra: la})
            elif how == "Inner":
                values, li, ri = np.intersect1d(
                    la.values, ra.values, return_indices=True
                )
                axis = _Axis(values)
                left, right = (
                    left.take(la, li, (axis,)),
                    right.take(ra, ri, (axis,)),
                )
            else:
                indexes, found = _indexes(ra, la.values)
                right = right.take(ra, indexes, (la,))
                right.present = _op("and", right.present, _mask(found, (la,)))
        elif how == "Inner" and ra is not None:
            indexes, found = _key_indexes(ra, lc.tensor)
            right = right.take(ra, indexes, lc.tensor.axes)
            right.present = _op(
                "and",
                right.present,
                lc.valid,
                _mask(found, lc.tensor.axes),
            )
        elif how == "Inner" and la is not None:
            indexes, found = _key_indexes(la, rc.tensor)
            left = left.take(la, indexes, rc.tensor.axes)
            left.present = _op(
                "and",
                left.present,
                rc.valid,
                _mask(found, rc.tensor.axes),
            )
        else:
            remaining.append(predicate)
    if how == "Left":
        if (
            any(len(a) != 1 for a in set(right.axes) - set(left.axes))
            or remaining
        ):
            raise _Decline(
                "outer join with free right axes or residual predicates"
            )
        columns = list(left.columns.items()) + [
            (k, _Column(c.tensor, _op("and", c.valid, right.present)))
            for k, c in right.columns.items()
        ]
        return _Table(_axes(left.axes, right.axes), columns, left.present)
    result = _Table(
        _axes(left.axes, right.axes),
        list(left.columns.items()) + list(right.columns.items()),
        _op("and", left.present, right.present),
    )
    for predicate in remaining:
        if isinstance(predicate, tuple):
            raise _Decline("dependent optimized join key")
        result = result.filter(_expr(predicate, result))
    return result


class _Bag:
    def __init__(self, tables):
        self.tables = [
            branch
            for table in tables
            for branch in (table.tables if isinstance(table, _Bag) else [table])
        ]

    def arrow(self, schema):
        return pa.concat_tables([t.arrow(schema) for t in self.tables])


def _map(table, fn):
    if isinstance(table, _Sorted):
        raise _Decline("ordered intermediate requires relational execution")
    return (
        _Bag([fn(t) for t in table.tables])
        if isinstance(table, _Bag)
        else fn(table)
    )


def _aggregate(table, groups, aggregates):
    keys = [_expr(g, table) for g in groups]
    dependent = [i for i, k in enumerate(keys) if k.tensor.axis is None]
    if dependent:
        if len(dependent) != 1 or any(not _is(k.valid, True) for k in keys):
            raise _Decline("aggregation by multiple dependent or nullable keys")
        return _segment(table, groups, aggregates, keys, dependent[0])
    if any(not _is(k.valid, True) for k in keys):
        raise _Decline("aggregation by a nullable key")
    kept = tuple(dict.fromkeys(k.tensor.axis for k in keys))
    columns = [(_key(g), k) for g, k in zip(groups, keys)]
    for expr in aggregates:
        raw = expr
        while raw.variant_name() == "Alias":
            raw = raw.to_variant().expr()
        agg = raw.to_variant()
        if agg.is_distinct():
            raise _Decline("DISTINCT aggregate")
        args = agg.args()
        col = _expr(args[0], table) if args else _column(1)
        columns.append(
            (
                _key(expr),
                _reduce(agg.aggregate_type().lower(), col, table, kept),
            )
        )
    # An ungrouped empty aggregate produces one row, with NULL SUM.
    present = (
        _op(">", _sum(_const(1), table.present, table.axes, kept), _const(0))
        if groups
        else _const(True)
    )
    return _Table(kept, columns, _truth(present))


def _aggregate_specs(aggregates):
    specs = []
    for expr in aggregates:
        raw = expr
        while raw.variant_name() == "Alias":
            raw = raw.to_variant().expr()
        agg = raw.to_variant()
        if agg.is_distinct():
            raise _Decline("DISTINCT aggregate")
        specs.append((_key(expr), agg.aggregate_type().lower(), agg.args()))
    return specs


def _segment(table, groups, aggregates, keys, dependent):
    key = keys[dependent].tensor
    values = key.get()
    runtime = _EXECUTION.get()
    if runtime is not None and runtime.owns(values):
        if key.domain is None:
            raise _Decline(
                "GPU grouping requires a proven finite integer domain"
            )
        unique = key.domain.values
        inverse, found = _indexes(key.domain, values)
        table = _Table(
            table.axes,
            table.columns.items(),
            _op("and", table.present, _mask(found, key.axes)),
        )
    else:
        if not values.size:
            raise _Decline("empty dependent group domain")
        if values.dtype.kind not in "iuUS":
            raise _Decline("dependent group key type")
        unique, inverse = np.unique(values, return_inverse=True)
    axis = _Axis(unique)
    independent = tuple(
        dict.fromkeys(
            k.tensor.axis for i, k in enumerate(keys) if i != dependent
        )
    )
    if not key.axes:
        columns = [
            (
                _key(g),
                _column(unique, (axis,), axis) if i == dependent else keys[i],
            )
            for i, g in enumerate(groups)
        ]
        for name, kind, args in _aggregate_specs(aggregates):
            col = _expr(args[0], table) if args else _column(1)
            columns.append((name, _reduce(kind, col, table, independent)))
        present = _op(
            ">",
            _sum(_const(1), table.present, table.axes, independent),
            _const(0),
        )
        return _Table(independent + (axis,), columns, _truth(present))
    local = _axes(independent, key.axes)
    output = independent + (axis,)
    shape = tuple(len(a) for a in local)
    result_shape = tuple(len(a) for a in output)
    rows = math.prod(len(a) for a in independent)
    indexes = (
        runtime.broadcast(inverse.reshape(values.shape), key.axes, local, shape)
        if runtime is not None and runtime.owns(inverse)
        else np.broadcast_to(
            _align(inverse.reshape(values.shape), key.axes, local), shape
        )
    ).reshape(rows, -1)

    def scatter(tensor):
        def compute():
            runtime = _EXECUTION.get()
            if runtime is not None and (
                runtime.owns(tensor.get()) or runtime.owns(indexes)
            ):
                data = runtime.broadcast(
                    tensor.get(), tensor.axes, local, shape
                ).reshape(rows, -1)
                return runtime.scatter(
                    data,
                    indexes,
                    len(unique),
                    count_bound=math.prod(len(a) for a in table.axes),
                ).reshape(result_shape)
            data = np.broadcast_to(
                _align(tensor.get(), tensor.axes, local), shape
            ).reshape(rows, -1)
            result = np.zeros((data.shape[0], len(unique)), dtype=data.dtype)
            np.add.at(
                result, (np.arange(data.shape[0])[:, None], indexes), data
            )
            return result.reshape(result_shape)

        result = _Tensor("reduce", output, value=compute)
        result.host_only = tensor.host_only and not (
            runtime is not None and runtime.owns(indexes)
        )
        if tensor.count_bound is not None:
            result.count_bound = tensor.count_bound * math.prod(
                len(a) for a in local if a not in independent
            )
        return result

    columns = [
        (_key(g), _column(unique, (axis,), axis) if i == dependent else keys[i])
        for i, g in enumerate(groups)
    ]
    for name, kind, args in _aggregate_specs(aggregates):
        if kind not in {"sum", "count", "avg"}:
            raise _Decline("dependent grouping aggregate")
        col = _expr(args[0], table) if args else _column(1)
        mask = _op("and", table.present, col.valid)
        count = scatter(_sum(_const(1), mask, table.axes, local))
        value_input = col.tensor
        if (
            kind == "avg"
            and getattr(value_input.get().dtype, "kind", "f") in "biu"
        ):
            value_input = _op("float64", value_input)
        value = (
            scatter(_sum(value_input, mask, table.axes, local))
            if kind != "count"
            else count
        )
        if kind == "avg":
            value = _op("/", value, count)
        columns.append(
            (
                name,
                _Column(
                    value,
                    _const(True)
                    if kind == "count"
                    else _truth(_op(">", count, _const(0))),
                ),
            )
        )
    present = _op(
        ">",
        scatter(_sum(_const(1), table.present, table.axes, local)),
        _const(0),
    )
    return _Table(output, columns, _truth(present))


def _group_flags(parts, keys, axes, key=None):
    masks = [
        p.present
        if key is None
        else _op("and", p.present, p.columns[key].valid)
        for p in parts
    ]
    sources = [tuple(p.columns[k].tensor.axis for k in keys) for p in parts]
    needed = set()
    for source, mask in zip(sources, masks):
        needed.update(source.index(a) for a in mask.axes)
        needed.update(
            i
            for i, (a, s) in enumerate(zip(axes, source))
            if not np.array_equal(a.values, s.values)
        )
    selected = tuple(sorted(needed))
    output = tuple(axes[i] for i in selected)

    def compute():
        runtime = _EXECUTION.get()
        if runtime is not None and any(
            runtime.owns(mask.get()) for mask in masks
        ):
            return (
                runtime.merge(
                    parts,
                    [keys[i] for i in selected],
                    key,
                    output,
                    flags="rows" if key is None else "valid",
                )
                > 0
            )
        result = np.zeros(tuple(len(a) for a in output), dtype=bool)
        for source, mask in zip(sources, masks):
            local = tuple(source[i] for i in selected)
            indexes = [
                np.searchsorted(a.values, s.values)
                for a, s in zip(output, local)
            ]
            values = np.broadcast_to(
                _align(mask.get(), mask.axes, local),
                tuple(len(a) for a in local),
            )
            if output:
                result[np.ix_(*indexes)] |= values
            else:
                result[...] |= values
        return result

    result = _Tensor("reduce", output, value=compute)
    result.host_only = all(m.host_only for m in masks)
    return _truth(result)


def _aggregate_bag(bag, groups, aggregates):
    specs = _aggregate_specs(aggregates)
    if any(kind not in {"sum", "count"} for _, kind, _ in specs):
        raise _Decline("UNION aggregate requires SUM or COUNT")
    parts = [_aggregate(t, groups, aggregates) for t in bag.tables]
    keys = [_key(g) for g in groups]
    if any(
        len({p.columns[k].tensor.axis for k in keys}) != len(keys)
        for p in parts
    ):
        raise _Decline("UNION grouping has correlated coordinate keys")
    axes = tuple(
        _Axis(
            np.unique(
                np.concatenate([p.columns[k].tensor.axis.values for p in parts])
            )
        )
        for k in keys
    )
    if (
        len({p.columns[k].tensor.axis for k in keys for p in parts}) == 0
        and groups
    ):
        raise _Decline("UNION group domain")
    shape = tuple(len(a) for a in axes)

    def merge(key, *, present=False):
        def compute():
            runtime = _EXECUTION.get()
            if (
                present
                and runtime is not None
                and any(runtime.owns(p.present.get()) for p in parts)
            ):
                return runtime.merge(parts, keys, key, axes, flags="rows")
            if (
                runtime is not None
                and not present
                and any(
                    runtime.owns(p.columns[key].tensor.get()) for p in parts
                )
            ):
                return runtime.merge(parts, keys, key, axes)
            dtype = (
                np.int64
                if present
                else parts[0].columns[key].tensor.get().dtype
            )
            result = np.zeros(shape, dtype=dtype)
            for part in parts:
                source_axes = tuple(part.columns[k].tensor.axis for k in keys)
                indexes = [
                    np.searchsorted(a.values, s.values)
                    for a, s in zip(axes, source_axes)
                ]
                col = (
                    part.columns[key]
                    if not present
                    else _Column(_const(1), _const(True))
                )
                mask = _op("and", part.present, col.valid)
                data = np.broadcast_to(
                    _align(col.tensor.get(), col.tensor.axes, source_axes),
                    tuple(len(a) for a in source_axes),
                )
                valid = np.broadcast_to(
                    _align(_numpy(mask.get()), mask.axes, source_axes),
                    data.shape,
                )
                if axes:
                    if all(
                        np.array_equal(a.values, s.values)
                        for a, s in zip(axes, source_axes)
                    ):
                        result += np.where(valid, data, 0)
                    else:
                        result[np.ix_(*indexes)] += np.where(valid, data, 0)
                else:
                    result[...] += np.where(valid, data, 0)
            return result

        return _Tensor("reduce", axes, value=compute)

    columns = [(k, _column(a.values, (a,), a)) for k, a in zip(keys, axes)]
    present = _group_flags(parts, keys, axes) if groups else _const(True)
    for key, kind, _ in specs:
        value = merge(key)
        valid = (
            _const(True)
            if kind == "count"
            else _group_flags(parts, keys, axes, key)
        )
        columns.append((key, _Column(value, valid)))
    return _Table(axes, columns, present)


class _Sorted:
    def __init__(self, table, sorts):
        self.table, self.sorts = table, sorts

    def arrow(self, schema):
        if isinstance(self.table, _Table):
            table = self.table
            ordered = []
            for expr in self.sorts:
                col = _expr(expr.expr(), table)
                axis = col.tensor.axis
                if axis is None or not _is(col.valid, True):
                    break
                if axis in ordered:
                    continue
                indexes = np.argsort(axis.values, kind="stable")
                if not expr.ascending():
                    indexes = indexes[::-1]
                if not np.array_equal(indexes, np.arange(len(axis))):
                    replacement = _Axis(axis.values[indexes])
                    table = table.take(axis, indexes, (replacement,))
                    axis = replacement
                ordered.append(axis)
            else:
                # Cartesian axes already define lexicographic row order.
                # Reorder compact dimensions before expanding coordinates.
                return _Table(
                    _axes(ordered, table.axes),
                    table.columns.items(),
                    table.present,
                ).arrow(schema)
        rows = self.table.arrow(schema)
        keys = []
        for expr in self.sorts:
            variant = expr
            key = _key(variant.expr())[1]
            if key not in rows.column_names:
                raise _Decline("sort key not present in output")
            keys.append(
                (key, "ascending" if variant.ascending() else "descending")
            )
        nulls = {s.nulls_first() for s in self.sorts}
        if len(nulls) > 1:
            raise _Decline("mixed sort NULL placement")
        return rows.take(
            pc.sort_indices(
                rows,
                sort_keys=keys,
                null_placement="at_start" if True in nulls else "at_end",
            )
        )


class DenseProgram:
    """Lower DataFusion logical operators to compact numerical values.

    Reads are supplied by the host for each invocation. Unsupported plans
    raise ``_Decline`` so the caller can execute its original SQL instead.
    """

    def __init__(self, plan, schema):
        self.plan, self.schema = plan, schema

    def run(self, read):
        def visit(plan):
            node = plan.to_variant()
            kind = type(node).__name__
            if kind == "TableScan":
                table = _map(
                    read(node.table_name()),
                    lambda t: t.alias(node.table_name()),
                )
                if isinstance(table, _Bag):
                    return table
                names = node.schema().field_names()
                by_name = {
                    f"{r}.{n}": (k, c)
                    for k, c in table.columns.items()
                    for r, n in [k]
                }
                table = _Table(
                    table.axes, [by_name[n] for n in names], table.present
                )
                for predicate in node.filters():
                    for part in _conjuncts(predicate):
                        table = table.filter(_expr(part, table))
                return table
            inputs = [visit(p) for p in plan.inputs()]
            if any(isinstance(t, _Sorted) for t in inputs):
                raise _Decline("nested ordering requires relational execution")
            if kind == "EmptyRelation":
                return _Table(present=_const(node.produce_one_row()))
            if kind == "SubqueryAlias":
                return _map(
                    inputs[0],
                    lambda t: t.alias(
                        node.alias(), node.schema().field_names()
                    ),
                )
            if kind == "Projection":
                return _map(
                    inputs[0],
                    lambda t: _Table(
                        t.axes,
                        [(_key(e), _expr(e, t)) for e in node.projections()],
                        t.present,
                    ),
                )
            if kind == "Filter":

                def apply_filter(table):
                    for part in _conjuncts(node.predicate()):
                        table = table.filter(_expr(part, table))
                    return table

                return _map(inputs[0], apply_filter)
            if kind in {"Join", "CrossJoin"}:
                if any(isinstance(t, _Bag) for t in inputs):
                    raise _Decline("join of UNION branches")
                predicates = []
                if kind == "Join":
                    # The SQL planner usually leaves keys in the filter.
                    predicates.extend(node.on())
                    if node.filter() is not None:
                        predicates.extend(_conjuncts(node.filter()))
                return _join(
                    *inputs,
                    predicates,
                    node.join_type() if kind == "Join" else "Inner",
                )
            if kind == "Aggregate":
                if isinstance(inputs[0], _Bag):
                    return _aggregate_bag(
                        inputs[0], node.group_by_exprs(), node.aggregate_exprs()
                    )
                return _aggregate(
                    inputs[0], node.group_by_exprs(), node.aggregate_exprs()
                )
            if kind == "Union":
                names = node.schema().field_names()
                return _Bag(
                    [
                        _map(
                            t,
                            lambda branch: _Table(
                                branch.axes,
                                [
                                    ((None, n), c)
                                    for n, c in zip(
                                        names, branch.columns.values()
                                    )
                                ],
                                branch.present,
                            ),
                        )
                        for t in inputs
                    ]
                )
            if kind == "Sort":
                if node.get_fetch_val() is not None:
                    raise _Decline("top-k sort")
                return _Sorted(inputs[0], node.sort_exprs())
            if kind == "WindowExpr":
                table = inputs[0]
                if isinstance(table, _Bag):
                    raise _Decline("window over UNION")
                columns = list(table.columns.items())
                for expr in node.get_window_expr():
                    if node.get_sort_exprs(
                        expr
                    ) or "UNBOUNDED FOLLOWING" not in str(node.get_frame(expr)):
                        raise _Decline("ordered or bounded window")
                    keys = [
                        _expr(k, table).tensor.axis
                        for k in node.get_partition_exprs(expr)
                    ]
                    if any(k is None for k in keys):
                        raise _Decline("dependent window key")
                    col = _expr(node.get_args(expr)[0], table)
                    columns.append(
                        (
                            _key(expr),
                            _reduce(
                                node.window_func_name(expr),
                                col,
                                table,
                                tuple(keys),
                            ),
                        )
                    )
                return _Table(table.axes, columns, table.present)
            raise _Decline(f"logical operator {kind}")

        return visit(self.plan)
