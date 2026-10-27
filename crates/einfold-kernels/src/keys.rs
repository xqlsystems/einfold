// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Rows' keys as bytes that are equal exactly when the host would call the
//! keys equal.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float16Type, Float32Type, Float64Type};
use arrow::row::{RowConverter, Rows, SortField};

use crate::{FloatKeys, Host, Result};

/// Each row's key for some columns, as bytes in Arrow's row format, which
/// gives equal keys equal bytes: NULL equals NULL, floats compare by their
/// bits, and a dictionary-encoded value encodes as the value itself. With no
/// columns, every key is empty. Pass columns through [`host_keys`] first.
pub(crate) struct Keys(Option<Rows>);

impl Keys {
    pub(crate) fn new(columns: &[ArrayRef]) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Keys(None));
        }
        let fields = columns
            .iter()
            .map(|c| SortField::new(c.data_type().clone()))
            .collect();
        Ok(Keys(Some(
            RowConverter::new(fields)?.convert_columns(columns)?,
        )))
    }

    pub(crate) fn get(&self, row: usize) -> &[u8] {
        self.0.as_ref().map_or(&[], |rows| rows.row(row).data())
    }
}

/// Column `$c`, of float type `$t`, with `-0.0` replaced by `0.0`, or `$c`
/// itself if it has none. `-0.0` is the one value whose bits are the sign
/// bit alone.
macro_rules! zero_signs {
    ($c:expr, $t:ty) => {{
        let values = $c.as_primitive::<$t>();
        let negative_zero =
            |v: <$t as ArrowPrimitiveType>::Native| v.to_bits() != 0 && v.to_bits() << 1 == 0;
        if values.values().iter().any(|&v| negative_zero(v)) {
            let zero = <$t as ArrowPrimitiveType>::Native::default();
            let fixed = values.unary::<_, $t>(|v| if negative_zero(v) { zero } else { v });
            Arc::new(fixed) as ArrayRef
        } else {
            Arc::clone($c)
        }
    }};
}

/// `columns` as the host compares them: with [`FloatKeys::SignedZerosEqual`],
/// each plain float column's `-0.0` becomes `0.0`. Dictionary-encoded floats
/// are left alone, as DataFusion leaves them.
pub(crate) fn host_keys(columns: &[ArrayRef], host: Host) -> Vec<ArrayRef> {
    match host.float_keys {
        FloatKeys::Bitwise => columns.to_vec(),
        FloatKeys::SignedZerosEqual => columns
            .iter()
            .map(|c| match c.data_type() {
                DataType::Float16 => zero_signs!(c, Float16Type),
                DataType::Float32 => zero_signs!(c, Float32Type),
                DataType::Float64 => zero_signs!(c, Float64Type),
                _ => Arc::clone(c),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    #[test]
    fn signed_zeros_are_one_key_only_when_the_host_says_so() {
        let col: ArrayRef = Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN]));
        let host = |float_keys| Host { float_keys };
        let same = |h: Host| {
            let keys = Keys::new(&host_keys(&[Arc::clone(&col)], h)).unwrap();
            keys.get(0) == keys.get(1)
        };
        assert!(same(host(FloatKeys::SignedZerosEqual)));
        assert!(!same(host(FloatKeys::Bitwise)));
    }
}
