// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S31: runs steps of a ddx gradient program again, from the tables
//! `profile.py` wrote, with einfold on and off, and prints each step's time
//! and einfold's `EXPLAIN ANALYZE`.
//!
//! ```text
//! einfold-spike-s31 DIR STEP.sql...
//! ```
//!
//! Every `*.parquet` file in `DIR` is registered as a table named after the
//! file, so a step's SQL reads the same rows it read in the program.

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::compute;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};

const REPEATS: usize = 3;

async fn session(einfold: bool, dir: &str) -> Result<SessionContext> {
    let builder = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new());
    let builder = if einfold {
        einfold_datafusion::install(builder)
    } else {
        builder
    };
    let ctx = SessionContext::new_with_state(builder.build());
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "parquet") {
            let name = path
                .file_stem()
                .expect("a file name")
                .to_string_lossy()
                .into_owned();
            // Read into memory, as ddxdb registers each step's rows.
            let frame = ctx
                .read_parquet(
                    path.to_string_lossy().as_ref(),
                    ParquetReadOptions::default(),
                )
                .await?;
            let schema = Arc::new(frame.schema().as_arrow().clone());
            let rows = frame.collect().await?;
            let rows = compute::concat_batches(&schema, &rows)?;
            ctx.register_table(
                &name,
                Arc::new(MemTable::try_new(schema, vec![vec![rows]])?),
            )?;
        }
    }
    Ok(ctx)
}

async fn best(ctx: &SessionContext, sql: &str) -> Result<(f64, usize)> {
    let mut best = f64::INFINITY;
    let mut rows = 0;
    for _ in 0..REPEATS {
        let start = Instant::now();
        let batches = ctx.sql(sql).await?.collect().await?;
        best = best.min(start.elapsed().as_secs_f64());
        rows = batches.iter().map(|b| b.num_rows()).sum();
    }
    Ok((best, rows))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, steps) = args.split_first().expect("DIR STEP.sql...");
    let on = session(true, dir).await?;
    let off = session(false, dir).await?;
    for step in steps {
        let sql = std::fs::read_to_string(step)?;
        let (with, rows) = best(&on, &sql).await?;
        let (without, _) = best(&off, &sql).await?;
        println!(
            "## {step}: {rows} rows, einfold {:.1} ms, DataFusion {:.1} ms",
            with * 1e3,
            without * 1e3
        );
        let explained = on
            .sql(&format!("EXPLAIN ANALYZE {sql}"))
            .await?
            .collect()
            .await?;
        println!(
            "{}",
            datafusion::arrow::util::pretty::pretty_format_batches(&explained)?
        );
    }
    Ok(())
}
