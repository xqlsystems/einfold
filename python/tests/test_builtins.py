# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""DataFusion's built-in functions answer alike with einfold installed and
without.

einfold plans every query in its own library, and runs DataFusion's own
copy of each built-in function there, rather than the session's copy
through the FFI (foreign function interface), whose every call converts the
session's whole configuration. The session's copies come from
datafusion-python's DataFusion, and einfold's from its own, which can be a
different release. So: for every deterministic scalar, aggregate and window
function the session lists, called with each of its signatures that sample
columns can fill, a session with einfold gives the same column types and
values, or fails with the same error, as one without. Functions whose
signatures are variadic get a menu of argument lists instead. Each call
runs with the session's default time zone and with `+05:00`, since
functions that read the session's configuration are where einfold's copies
could differ (einfold#136).

Every sample column holds a NULL in its last row, and text columns hold
timestamps, date parts, words and numbers, so that each function meets
input it accepts and input it rejects."""

from __future__ import annotations

import datetime as dt
import decimal
import math
from collections import defaultdict

import pyarrow as pa
import pytest
from datafusion import SessionConfig, SessionContext

from einfold.datafusion import EinFold

_TIMES = [dt.datetime(2024, 3, 4, 5, 6, 7), dt.datetime(1999, 12, 31, 23, 59, 58),
          dt.datetime(2024, 2, 29, 0, 0, 1)]


def _column(values, ty):
    return pa.array(values + [None], ty)


def _typed_columns() -> dict[str, pa.Array]:
    """A sample column for each type `information_schema.parameters` names."""
    integers = [0, 2, 7]
    columns = {name: _column(integers, ty) for name, ty in (
        ("Int8", pa.int8()), ("Int16", pa.int16()), ("Int32", pa.int32()),
        ("Int64", pa.int64()), ("UInt8", pa.uint8()), ("UInt16", pa.uint16()),
        ("UInt32", pa.uint32()), ("UInt64", pa.uint64()))}
    columns |= {
        "Float32": _column([0.5, -2.25, 3.0], pa.float32()),
        "Float64": _column([0.5, -2.25, 3.0], pa.float64()),
        "Boolean": _column([True, False, True], pa.bool_()),
        "Binary": _column([b"ab", b"", b"\x00\xff"], pa.binary()),
        "Date": _column([t.date() for t in _TIMES], pa.date32()),
        "Decimal(38, 10)": _column(
            [decimal.Decimal("1.5"), decimal.Decimal("-2.25"), decimal.Decimal("0")],
            pa.decimal128(38, 10)),
        "Interval(MonthDayNano)": _column(
            [pa.MonthDayNano([1, 2, 3]), pa.MonthDayNano([0, 0, 10**9]),
             pa.MonthDayNano([0, 1, 0])], pa.month_day_nano_interval()),
    }
    for unit, name in (("s", "s"), ("ms", "ms"), ("us", "µs"), ("ns", "ns")):
        columns[f"Timestamp({name})"] = _column(_TIMES, pa.timestamp(unit))
        columns[f'Timestamp({name}, "+TZ")'] = _column(_TIMES, pa.timestamp(unit, "+02:00"))
        columns[f"Duration({name})"] = _column(
            [dt.timedelta(seconds=90), dt.timedelta(0), dt.timedelta(days=2)],
            pa.duration(unit))
        time = pa.time32(unit) if unit in ("s", "ms") else pa.time64(unit)
        columns[f"Time({name})"] = _column(
            [dt.time(5, 6, 7), dt.time(0, 0, 0), dt.time(23, 59, 59)], time)
    return columns


_TYPED = _typed_columns()
_COLUMN_OF = {ty: f"c{i}" for i, ty in enumerate(_TYPED)}

# Text in four kinds, each tried for a signature's every text argument.
_TEXT = {
    "s_time": ["2024-03-04 05:06:07", "1999-12-31T23:59:58Z", "2024-02-29"],
    "s_part": ["year", "month", "day"],
    "s_word": ["a,b,c", "Hello World", ""],
    "s_number": ["12", "-3.5", "0"],
}

# Argument lists for functions whose signatures are variadic, which
# `information_schema.parameters` doesn't list.
_VARIADIC = [
    [], ["t.s_time"], ["t.s_word"], ["t.s_number"], ["t.s_time", "t.s_format"],
    ["t.s_word", "t.s_word"], ["t.s_word", "t.s_word", "t.s_word"],
    [f"t.{_COLUMN_OF['Int64']}"], [f"t.{_COLUMN_OF['Float64']}"],
    [f"t.{_COLUMN_OF['Int64']}", f"t.{_COLUMN_OF['Int64']}"],
    [f"t.{_COLUMN_OF['Float64']}", f"t.{_COLUMN_OF['Float64']}"],
    [f"t.{_COLUMN_OF['Timestamp(ns)']}"], [f"t.{_COLUMN_OF['Date']}"],
]


def _table() -> pa.RecordBatch:
    columns = {_COLUMN_OF[ty]: a for ty, a in _TYPED.items()}
    columns |= {name: _column(values, pa.string()) for name, values in _TEXT.items()}
    columns["s_format"] = _column(["%Y-%m-%d %H:%M:%S"] * 3, pa.string())
    columns["g"] = pa.array([0, 0, 1, 1], pa.int64())
    columns["k"] = pa.array([1, 2, 3, 4], pa.int64())
    return pa.record_batch(list(columns.values()), names=list(columns))


def _session(einfold: bool, time_zone: str | None) -> SessionContext:
    ctx = SessionContext(SessionConfig().with_information_schema(True))
    if einfold:
        ctx.with_extensions(EinFold())
    if time_zone:
        ctx.sql(f"SET datafusion.execution.time_zone = '{time_zone}'").collect()
    ctx.register_record_batches("t", [[_table()]])
    return ctx


def _call(name: str, kind: str, args: list[str]) -> str:
    f = f'"{name}"({", ".join(args)})'
    if kind == "AGGREGATE":
        return f"SELECT t.g, {f} AS x FROM t GROUP BY t.g ORDER BY t.g"
    if kind == "WINDOW":
        return f"SELECT t.k, {f} OVER (PARTITION BY t.g ORDER BY t.k) AS x FROM t ORDER BY t.k"
    return f"SELECT t.k, {f} AS x FROM t ORDER BY t.k"


def _calls(ctx: SessionContext):
    """(function, SQL) for each deterministic built-in, once per argument
    list its signatures allow."""
    signatures = defaultdict(lambda: defaultdict(dict))
    for p in ctx.sql("SELECT specific_name, rid, ordinal_position, data_type "
                     "FROM information_schema.parameters "
                     "WHERE parameter_mode = 'IN'").to_arrow_table().to_pylist():
        signatures[p["specific_name"]][p["rid"]][p["ordinal_position"]] = p["data_type"]
    kinds = {r["routine_name"]: r["function_type"] for r in ctx.sql(
        "SELECT DISTINCT routine_name, function_type FROM information_schema.routines "
        "WHERE is_deterministic AND function_type <> 'TABLE'").to_arrow_table().to_pylist()}
    for name, kind in sorted(kinds.items()):
        lists = []
        for params in signatures[name].values():
            types = [params[i] for i in sorted(params)]
            if not all(ty in _COLUMN_OF or ty in ("String", "Null") for ty in types):
                continue
            kinds_of_text = _TEXT if "String" in types else [None]
            for text in kinds_of_text:
                lists.append([f"t.{text}" if ty == "String" else
                              "NULL" if ty == "Null" else f"t.{_COLUMN_OF[ty]}"
                              for ty in types])
        if not signatures[name]:
            lists = _VARIADIC
        seen = set()
        for args in lists:
            if tuple(args) not in seen:
                seen.add(tuple(args))
                yield name, _call(name, kind, args)


def _answer(ctx: SessionContext, sql: str):
    """The column types and values `sql` gives, or its error. An error that
    crosses the FFI says so; the rest of its message is compared."""
    try:
        table = ctx.sql(sql).to_arrow_table()
    except Exception as e:  # noqa: BLE001 — any error is an answer
        return ("error", str(e).replace("FFI error: ", "").splitlines()[0])
    return ("ok", [f.type for f in table.schema],
            [_values(table.column(i)) for i in range(table.num_columns)])


def _values(column: pa.ChunkedArray) -> list:
    """A column's values; times as their integers, which Python's `datetime`
    can't hold at nanoseconds."""
    if pa.types.is_temporal(column.type) and not pa.types.is_interval(column.type):
        width = pa.int32() if pa.types.is_date32(column.type) or pa.types.is_time32(
            column.type) else pa.int64()
        column = column.cast(width)
    return column.to_pylist()


def _same(a, b) -> bool:
    """Equal, with every NaN alike."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


# Differences with an open issue: (time zone, function).
_KNOWN = {("+05:00", "to_unixtime")}  # einfold#136


@pytest.mark.parametrize("time_zone", [None, "+05:00"])
def test_builtins_answer_alike(time_zone):
    off, on = _session(False, time_zone), _session(True, time_zone)
    differences, answered = [], set()
    for name, sql in _calls(off):
        want, got = _answer(off, sql), _answer(on, sql)
        if want[0] == "ok":
            answered.add(name)
        if (time_zone, name) not in _KNOWN and not _same(want, got):
            differences.append(f"{sql}\n  without einfold: {want}\n  with einfold:    {got}")
    assert not differences, "\n".join(differences)
    # So that a change to the sample columns, or to what the session lists,
    # can't make this pass by calling nothing.
    assert len(answered) >= 120, sorted(answered)


@pytest.mark.xfail(strict=True, reason="einfold#136: functions planned in einfold "
                   "run with DataFusion's default configuration (einfold#137)")
def test_to_unixtime_ignores_the_sessions_time_zone():
    """Pins einfold#136, and through it datafusion-ffi's gap (einfold#137):
    fails once `to_unixtime` reads text in the session's time zone."""
    sql = "SELECT t.k, to_unixtime(t.s_time) AS x FROM t ORDER BY t.k"
    off, on = _session(False, "+05:00"), _session(True, "+05:00")
    assert _same(_answer(off, sql), _answer(on, sql))
