-- SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
--
-- SPDX-License-Identifier: Apache-2.0

-- Spike S1/S13/S14 probe for zarr-datafusion's zarr-cli.
-- Run from this folder: zarr-cli -f probe_zarr_datafusion.sql
CREATE EXTERNAL TABLE ds STORED AS ZARR LOCATION 'fixture/v3.zarr';
DESCRIBE ds;
SELECT count(*) AS rows,
       sum(CASE WHEN t2m IS NULL THEN 1 ELSE 0 END) AS t2m_null,
       sum(CASE WHEN isnan(t2m) THEN 1 ELSE 0 END) AS t2m_nan,
       sum(CASE WHEN x = 0 THEN 1 ELSE 0 END) AS x_zero
FROM ds;
SET datafusion.explain.show_statistics = true;
EXPLAIN SELECT * FROM ds;
EXPLAIN ANALYZE SELECT count(*) FROM ds WHERE lat > 10;
