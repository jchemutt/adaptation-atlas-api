# Atlas Hazard Exposure Query API (FastAPI + Redis + DuckDB materialize)

This is a small server-side query service designed to make **Quarto OJS notebooks fast** when the raw Parquet datasets are very large.

**Key idea**

- The notebook sends filters (scenario/timeframe/geo/hazard_vars/commodities)
- The API runs the heavy query server-side (DuckDB)
- The API returns **small chart-ready JSON**
- Results are cached:
  - **Redis** = hot cache (fastest, TTL)
  - **DuckDB file** = _materialized cache_ that persists across restarts

---

## Quick start (Docker)

1. Open a terminal in this folder
2. Run:

```bash
docker compose up --build
```

API should be available at:

- `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

The materialized DuckDB cache is persisted in `./data/materialized_cache.duckdb`.

The local Parquet cache is persisted in `./data/parquet_cache`.

---

## Prefetching Parquet files

This project can also warm the **local Parquet cache** ahead of time using `prefetch_parquets.py`.

What it does:

- downloads the known public Parquet files used by the frontend into the local cache directory
- uses the same hash-based cache filename logic as the API, so prefetched files are recognized as already downloaded
- skips files that already exist unless you force a refresh

This is useful when you want to reduce slow first-time requests caused by downloading Parquet files on demand from public S3.

### Run the prefetch script inside Docker

```bash
docker compose exec api python /app/prefetch_parquets.py
```

### Optional variants

Verify what is already cached without downloading:

```bash
docker compose exec api python /app/prefetch_parquets.py --verify
```

Force a fresh re-download of all prefetched files:

```bash
docker compose exec api python /app/prefetch_parquets.py --force
```

Use a different worker count:

```bash
docker compose exec api python /app/prefetch_parquets.py --workers 4
```

## Cache behavior summary

There are two different cache layers in this service:

### 1. Response cache

Used for API responses after a query has already been computed.

- Redis stores hot JSON responses
- DuckDB stores materialized JSON responses that survive restarts

### 2. Local Parquet cache

Used for raw Parquet source files before query execution.

- when a requested Parquet is already present locally, the API reads it from `/data/parquet_cache`
- when it is missing locally, the API downloads it from its public HTTPS source and stores it in the local cache
- the prefetch script helps warm this cache before users hit the API

---

## Notes

- `./data` is mounted into the container as `/data`, so both the materialized DuckDB file and the Parquet cache survive container restarts
- if you move the Parquet cache to another compatible server, the app can reuse those files as long as the hash naming logic stays the same
