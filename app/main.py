import os
import json
import time
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import duckdb
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis


# ----------------------------
# Settings
# ----------------------------

@dataclass
class Settings:
    redis_url: str
    duckdb_path: str
    duckdb_threads: int
    cache_ttl_seconds: int
    cache_clear_token: str
    cache_clear_local_only: bool
    cache_materialize: bool
    materialize_keep_days: int

    allow_any_url: bool
    allowed_parquet_hosts: List[str]
    allow_s3_urls: bool
    allowed_s3_buckets: List[str]

    cors_origins: List[str]
    cors_origin_regex: Optional[str]
    allow_broad_geo: bool

    parquet_magic_check: bool
    export_max_rows: int

    @classmethod
    def from_env(cls) -> "Settings":
        def _bool(name: str, default: str) -> bool:
            return os.getenv(name, default).strip().lower() == "true"

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        duckdb_path = os.getenv("DUCKDB_DB_PATH", "/data/materialized_cache.duckdb")
        duckdb_threads = int(os.getenv("DUCKDB_THREADS", "8"))
        cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
        cache_clear_token = os.getenv("CACHE_CLEAR_TOKEN", "").strip()
        cache_clear_local_only = _bool("CACHE_CLEAR_LOCAL_ONLY", "true")
        cache_materialize = _bool("CACHE_MATERIALIZE", "true")
        materialize_keep_days = int(os.getenv("MATERIALIZE_KEEP_DAYS", "30"))

        allow_any_url = _bool("ALLOW_ANY_URL", "false")
        allowed_parquet_hosts = [
            h.strip().lower()
            for h in os.getenv("ALLOWED_PARQUET_HOSTS", "digital-atlas.s3.amazonaws.com").split(",")
            if h.strip()
        ]


        allow_s3_urls = _bool("ALLOW_S3_URLS", "true")
        allowed_s3_buckets = [b.strip() for b in os.getenv("ALLOWED_S3_BUCKETS", "digital-atlas").split(",") if b.strip()]
        cors_origins = [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:4774,http://127.0.0.1:4774,http://localhost:8000,http://127.0.0.1:8000",
            ).split(",")
            if o.strip()
        ]

        # Optional: allow origin regex (useful for Quarto preview random ports in dev)
        cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", "").strip() or r"^https?://(localhost|127\.0\.0\.1)(:\\d+)?$"

        allow_broad_geo = _bool("ALLOW_BROAD_GEO", "false")

        parquet_magic_check = _bool("PARQUET_MAGIC_CHECK", "true")
        export_max_rows = int(os.getenv("EXPORT_MAX_ROWS", "200000"))

        return cls(
            redis_url=redis_url,
            duckdb_path=duckdb_path,
            duckdb_threads=duckdb_threads,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_clear_token=cache_clear_token,
            cache_clear_local_only=cache_clear_local_only,
            cache_materialize=cache_materialize,
            materialize_keep_days=materialize_keep_days,
            allow_any_url=allow_any_url,
            allowed_parquet_hosts=allowed_parquet_hosts,
            allow_s3_urls=allow_s3_urls,
            allowed_s3_buckets=allowed_s3_buckets,
            cors_origins=cors_origins,
            cors_origin_regex=cors_origin_regex,
            allow_broad_geo=allow_broad_geo,
            parquet_magic_check=parquet_magic_check,
            export_max_rows=export_max_rows,
        )


S = Settings.from_env()

print('CORS config:', {'cors_origins': S.cors_origins, 'cors_origin_regex': S.cors_origin_regex})



# ----------------------------
# App
# ----------------------------

app = FastAPI(title="Atlas Hazard Exposure Query API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # If you set CORS_ORIGIN_REGEX, it takes precedence and allows matching origins (e.g., any localhost port).
        # NOTE: allow_origin_regex (if set) will also be honored by Starlette.
    allow_origins=(['*'] if ('*' in S.cors_origins) else S.cors_origins),
    allow_origin_regex=S.cors_origin_regex,
    # We don't use cookies/auth from the browser; credentials can stay off (also avoids '*' + credentials issues).
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis: Optional[Redis] = None


# ----------------------------
# Request models
# ----------------------------

class GeoFilter(BaseModel):
    admin0: List[str] = Field(default_factory=list, description="admin0_name values; use ['all'] for broad")
    admin1: List[str] = Field(default_factory=list, description="admin1_name values")
    admin2: List[str] = Field(default_factory=list, description="admin2_name values")


class ScenarioPick(BaseModel):
    scenario: str
    timeframe: str


class BaseQuery(BaseModel):
    dataset_url: str = Field(..., description="HTTPS URL to hazard-exposure parquet (interaction.parquet)")
    scen: ScenarioPick
    geo: GeoFilter

    commodities: List[str] = Field(default_factory=list, description="crop codes; use ['all'] for all")
    hazard_vars: Optional[List[str]] = Field(default=None, description="hazard_vars values")

    # Used only if hazard_vars is not provided.
    method: str = Field(default="generic", description="generic | crop_specific")
    commodity_group: str = Field(default="all")

    cache_ttl_seconds: Optional[int] = None


class TotalsByHazardRequest(BaseQuery):
    hazards: Optional[List[str]] = None


class TotalsByCropRequest(BaseQuery):
    hazards: Optional[List[str]] = None


class HazardByCropRequest(BaseQuery):
    # Return a hazard×crop matrix (already aggregated), suitable for stacked bars or heatmaps.
    # Optional limits keep payloads small & UI snappy.
    hazards: Optional[List[str]] = None
    top_hazards: Optional[int] = None
    top_crops: Optional[int] = None
    bucket_other: Optional[bool] = Field(
        default=True,
        description='If true, bucket non-top crops into "Other" when top_crops is set. If false, drop non-top crops instead.',
    )


class ByAdminRequest(BaseQuery):
    group_child: bool = True
    group_hazard: bool = False
    hazards: Optional[List[str]] = None


class DenomTotalRequest(BaseModel):
    denom_url: str = Field(..., description="HTTPS or s3:// URL to total exposure parquet")
    geo: GeoFilter
    commodities: List[str] = Field(default_factory=list)
    # Shiny parity: exposure + unit are separate.
    exposure: Optional[str] = None
    unit: Optional[str] = None
    # Backwards compatibility: some clients send only exposure_unit.
    exposure_unit: Optional[str] = None
    cache_ttl_seconds: Optional[int] = None


class Q1Request(BaseModel):
    left: TotalsByHazardRequest
    right: TotalsByHazardRequest
    denom: Optional[DenomTotalRequest] = None


class Q2Request(BaseModel):
    left: HazardByCropRequest
    right: HazardByCropRequest
    denom: Optional[DenomTotalRequest] = None


class RecordsRequest(BaseQuery):
    page: int = 1
    page_size: int = 100
    sort: str = Field(default="value_desc", description="value_desc | value_asc")


# ----------------------------
# Utilities
# ----------------------------

def _ttl(req_ttl: Optional[int]) -> Optional[int]:
    """Return cache TTL semantics.

    - None  => use server default (S.cache_ttl_seconds)
    - <0    => disable cache for this request (no read/write)
    - 0     => no expiry (persistent until manually cleared)
    - >0    => expiry in seconds
    """
    v: int
    if req_ttl is None:
        v = int(S.cache_ttl_seconds)
    else:
        try:
            v = int(req_ttl)
        except Exception:
            v = int(S.cache_ttl_seconds)

    if v < 0:
        return -1
    if v == 0:
        return None
    return max(1, v)


def _sha1_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_key(prefix: str, payload: Any) -> str:
    return f"{prefix}:{_sha1_json(payload)}"


def _sql_q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _norm_list(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        if v is None:
            continue
        vv = str(v).strip()
        if vv:
            out.append(vv)
    return out


def _is_broad_geo(geo: GeoFilter) -> bool:
    a0 = _norm_list(geo.admin0)
    a1 = _norm_list(geo.admin1)
    a2 = _norm_list(geo.admin2)
    all0 = (len(a0) == 0) or ("all" in [x.lower() for x in a0])
    return all0 and len(a1) == 0 and len(a2) == 0


def _geo_where(geo: GeoFilter) -> str:
    admin0 = _norm_list(geo.admin0)
    admin1 = _norm_list(geo.admin1)
    admin2 = _norm_list(geo.admin2)

    # "all" in admin0 means no admin0 filter.
    has_all = (len(admin0) == 0) or any(x.lower() == "all" for x in admin0)
    a0 = [] if has_all else [x for x in admin0 if x.lower() != "all"]
    a1 = [] if has_all else admin1
    a2 = [] if has_all else admin2

    wh: List[str] = []
    if len(a0) > 0:
        wh.append(f"admin0_name IN ({', '.join(_sql_q(v) for v in a0)})")

    if len(a2) > 0:
        if len(a1) > 0:
            wh.append(f"admin1_name IN ({', '.join(_sql_q(v) for v in a1)})")
        wh.append(f"admin2_name IN ({', '.join(_sql_q(v) for v in a2)})")
    elif len(a1) > 0:
        wh.append(f"admin1_name IN ({', '.join(_sql_q(v) for v in a1)})")
        wh.append("admin2_name IS NULL")
    else:
        wh.append("admin1_name IS NULL")
        wh.append("admin2_name IS NULL")

    return " AND ".join(wh) if wh else "TRUE"





def _geo_where_parent(geo: GeoFilter) -> str:
    """Looser geo filter used when we want *child breakdown* (by-admin).

    Unlike _geo_where(), this does NOT force admin1_name/admin2_name to be NULL
    when admin1/admin2 are not provided. It only applies the explicit selections.
    """
    admin0 = _norm_list(geo.admin0)
    admin1 = _norm_list(geo.admin1)
    admin2 = _norm_list(geo.admin2)

    # "all" in admin0 means no admin0 filter (and treat as broad).
    has_all = (len(admin0) == 0) or any(x.lower() == "all" for x in admin0)
    a0 = [] if has_all else [x for x in admin0 if x.lower() != "all"]
    a1 = [] if has_all else admin1
    a2 = [] if has_all else admin2

    wh: List[str] = []
    if len(a0) > 0:
        wh.append(f"admin0_name IN ({', '.join(_sql_q(v) for v in a0)})")
    if len(a1) > 0:
        wh.append(f"admin1_name IN ({', '.join(_sql_q(v) for v in a1)})")
    if len(a2) > 0:
        wh.append(f"admin2_name IN ({', '.join(_sql_q(v) for v in a2)})")

    return " AND ".join(wh) if wh else "TRUE"

def _scen_where(scen: ScenarioPick) -> str:
    sc = str(scen.scenario).strip()
    tf = str(scen.timeframe).strip()
    if not sc or not tf:
        return "FALSE"
    return f"scenario = {_sql_q(sc)} AND timeframe = {_sql_q(tf)}"


def _crop_where(commodities: List[str]) -> str:
    vals = _norm_list(commodities)
    if len(vals) == 0:
        return "TRUE"
    if any(v.lower() == "all" for v in vals):
        vals = [v for v in vals if v.lower() != "all"]
        if len(vals) == 0:
            return "TRUE"
    return f"crop IN ({', '.join(_sql_q(v) for v in vals)})"


def _haz_where(hazards: Optional[List[str]]) -> str:
    vals = _norm_list(hazards or [])
    if len(vals) == 0:
        return "TRUE"
    return f"hazard IN ({', '.join(_sql_q(v) for v in vals)})"


def _hazard_vars_where(hazard_vars: Optional[List[str]], method: str, commodity_group: str) -> str:
    # If user provides hazard_vars, use them.
    if hazard_vars is not None:
        vals = _norm_list(hazard_vars)
        if len(vals) == 0:
            return "TRUE"
        return f"hazard_vars IN ({', '.join(_sql_q(v) for v in vals)})"

    # Defaults (matching what you used in the notebook)
    generic = ["NDWS+NTx35+NDWL0", "NDWS+THI-max+NDWL0"]
    crop_specific = ["PTOT-L+NTxS+PTOT-G", "PTOT-L+THI-max+PTOT-G"]

    m = (method or "").lower().strip()
    vals = crop_specific if m in ("crop", "crop_specific", "crop-specific") else generic
    return f"hazard_vars IN ({', '.join(_sql_q(v) for v in vals)})"


def _validate_url(url: str) -> None:
    # Allow opt-out of URL restrictions for trusted deployments.
    if S.allow_any_url:
        return

    u = urlparse(url)
    scheme = (u.scheme or "").lower()

    if scheme == "https":
        host = (u.hostname or "").lower()
        if host not in S.allowed_parquet_hosts:
            raise HTTPException(
                status_code=400,
                detail=f"Host '{host}' not allowlisted. Allowed: {', '.join(S.allowed_parquet_hosts)}",
            )
        return

    if scheme == "s3":
        if not S.allow_s3_urls:
            raise HTTPException(status_code=400, detail="s3:// URLs are disabled")
        bucket = (u.netloc or "").strip()
        if not bucket:
            raise HTTPException(status_code=400, detail="Invalid s3:// URL (missing bucket)")
        if bucket not in S.allowed_s3_buckets:
            raise HTTPException(
                status_code=400,
                detail=f"S3 bucket '{bucket}' not allowlisted. Allowed: {', '.join(S.allowed_s3_buckets)}",
            )
        return

    raise HTTPException(status_code=400, detail="Only https:// or s3:// URLs are allowed")


def _parquet_magic_check(url: str) -> None:
    if not S.parquet_magic_check:
        return
    # Magic header check only works for HTTP(S) URLs.
    try:
        if urlparse(url).scheme.lower() != "https":
            return
    except Exception:
        return
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            r = client.get(url, headers={"Range": "bytes=0-3"})
            if r.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"Parquet URL returned {r.status_code}")
            if r.content != b"PAR1":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "URL did not look like a parquet file (missing PAR1 header). "
                        "This often happens when the URL is wrong or access is denied."
                    ),
                )
    except HTTPException:
        raise
    except Exception:
        # Best-effort only
        return


def _duckdb_connect(for_http_parquet: bool = False) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(S.duckdb_path), exist_ok=True)
    con = duckdb.connect(S.duckdb_path)
    con.execute(f"PRAGMA threads={S.duckdb_threads}")
    con.execute("PRAGMA enable_object_cache=true")
    con.execute("SET preserve_insertion_order=false")

    if for_http_parquet:
        try:
            con.execute("LOAD httpfs")
        except Exception:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")

    return con


# ----------------------------
# Materialized cache (DuckDB) + Redis
# ----------------------------

class CacheStore:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def init_materialized_cache(self) -> None:
        if not S.cache_materialize:
            return
        con = _duckdb_connect(for_http_parquet=False)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                  cache_key VARCHAR PRIMARY KEY,
                  response_json VARCHAR,
                  created_at TIMESTAMP
                );
                """
            )
            # cleanup old entries
            if S.materialize_keep_days > 0:
                # DuckDB does not support parameter placeholders in INTERVAL.
                # This value comes from server config (not user input), so safe to inline.
                keep_days = int(S.materialize_keep_days)
                con.execute(
                    f"DELETE FROM response_cache WHERE created_at < (now() - INTERVAL '{keep_days} days')"
                )
        finally:
            con.close()

    async def get_json(self, key: str, ttl_seconds: Optional[int]) -> Tuple[Optional[Any], str]:
        # ttl_seconds == -1 means cache disabled for this request
        if ttl_seconds == -1:
            return None, "disabled"

        # 1) Redis hit
        cached = await self.redis.get(key)
        if cached:
            try:
                return json.loads(cached), "redis"
            except Exception:
                # fall through
                pass

        # 2) DuckDB materialized cache hit
        if not S.cache_materialize:
            return None, "miss"

        con = _duckdb_connect(for_http_parquet=False)
        try:
            row = con.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?",
                [key],
            ).fetchone()
        finally:
            con.close()

        if not row:
            return None, "miss"

        raw = row[0]
        # refresh Redis (so next request can hit Redis quickly)
        if ttl_seconds is None or (isinstance(ttl_seconds, int) and ttl_seconds <= 0):
            await self.redis.set(key, raw)
        else:
            await self.redis.set(key, raw, ex=int(ttl_seconds))

        try:
            return json.loads(raw), "duckdb"
        except Exception:
            return None, "miss"

    async def set_json(self, key: str, value: Any, ttl_seconds: Optional[int]) -> None:
        raw = json.dumps(value, separators=(",", ":"))

        # ttl_seconds semantics:
        #  -1   => do not write cache
        #  None => persist without expiry
        #  >0   => expire in seconds
        if ttl_seconds == -1:
            return
        if ttl_seconds is None:
            await self.redis.set(key, raw)
        else:
            # Redis rejects EX=0 / negative; guard here just in case a caller bypassed _ttl()
            if int(ttl_seconds) <= 0:
                await self.redis.set(key, raw)
            else:
                await self.redis.set(key, raw, ex=int(ttl_seconds))

        # materialize to DuckDB
        if not S.cache_materialize:
            return

        con = _duckdb_connect(for_http_parquet=False)
        try:
            con.execute("DELETE FROM response_cache WHERE cache_key = ?", [key])
            con.execute(
                "INSERT INTO response_cache VALUES (?, ?, now())",
                [key, raw],
            )
        finally:
            con.close()



    async def clear_prefixes(self, prefixes: List[str], *, dry_run: bool = False, batch_size: int = 1000) -> Dict[str, Any]:
        """Delete cached responses for one or more key prefixes (e.g., 'by_admin', 'q1').

        Uses SCAN + UNLINK (non-blocking delete) in batches.
        """
        deleted_total = 0
        patterns = [f"{p}:*" for p in prefixes if p]
        details: Dict[str, int] = {p: 0 for p in prefixes if p}

        for p, pattern in zip([p for p in prefixes if p], patterns):
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=batch_size)
                if keys:
                    if not dry_run:
                        # UNLINK is preferred (non-blocking). Fallback to DEL if needed.
                        try:
                            n = await self.redis.unlink(*keys)
                        except Exception:
                            n = await self.redis.delete(*keys)
                        details[p] += int(n or 0)
                        deleted_total += int(n or 0)
                    else:
                        details[p] += len(keys)
                        deleted_total += len(keys)

                if cursor == 0:
                    break

        return {"deleted": deleted_total, "by_prefix": details, "dry_run": dry_run}

cache_store: Optional[CacheStore] = None


# ----------------------------
# Core query functions
# ----------------------------

def _rows(con: duckdb.DuckDBPyConnection, query: str) -> List[Dict[str, Any]]:
    rel = con.execute(query)
    cols = [d[0] for d in rel.description]
    out: List[Dict[str, Any]] = []
    for r in rel.fetchall():
        out.append({cols[i]: r[i] for i in range(len(cols))})
    return out


def _query_totals_by_hazard(req: TotalsByHazardRequest) -> List[Dict[str, Any]]:
    _validate_url(req.dataset_url)
    _parquet_magic_check(req.dataset_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection (admin0=all with no admin1/admin2) is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )

    geo_where_expr = _geo_where(req.geo)

    q = f"""
      SELECT hazard, COALESCE(SUM(CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE) ELSE NULL END), 0.0) AS total
      FROM read_parquet({_sql_q(req.dataset_url)})
      WHERE {_scen_where(req.scen)}
        AND {geo_where_expr}
        AND {_crop_where(req.commodities)}
        AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
        AND {_haz_where(req.hazards)}
      GROUP BY hazard
      ORDER BY total DESC
    """

    con = _duckdb_connect(for_http_parquet=True)
    try:
        return _rows(con, q)
    finally:
        con.close()


def _query_totals_by_crop(req: TotalsByCropRequest) -> List[Dict[str, Any]]:
    _validate_url(req.dataset_url)
    _parquet_magic_check(req.dataset_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )

    geo_where_expr = _geo_where(req.geo)

    q = f"""
      SELECT crop, COALESCE(SUM(CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE) ELSE NULL END), 0.0) AS total
      FROM read_parquet({_sql_q(req.dataset_url)})
      WHERE {_scen_where(req.scen)}
        AND {geo_where_expr}
        AND {_crop_where(req.commodities)}
        AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
        AND {_haz_where(req.hazards)}
      GROUP BY crop
      ORDER BY total DESC
    """

    con = _duckdb_connect(for_http_parquet=True)
    try:
        return _rows(con, q)
    finally:
        con.close()


def _resolve_admin_group_fields(geo: GeoFilter) -> Tuple[str, str]:
    a1 = _norm_list(geo.admin1)
    a2 = _norm_list(geo.admin2)

    if len(a2) > 0:
        return ("admin2_name", "TRUE")
    if len(a1) > 0:
        return ("admin2_name", "admin2_name IS NOT NULL")
    return ("admin1_name", "admin1_name IS NOT NULL")




def _resolve_admin_group_fields_current(geo: GeoFilter) -> Tuple[str, str]:
    """Return grouping field for *current* selected level (not children).

    Examples:
      - geo.admin0 set, geo.admin1 empty -> group by admin0_name
      - geo.admin1 set, geo.admin2 empty -> group by admin1_name
      - geo.admin2 set -> group by admin2_name
    """
    a1 = _norm_list(geo.admin1)
    a2 = _norm_list(geo.admin2)

    if len(a2) > 0:
        return ("admin2_name", "admin2_name IS NOT NULL")
    if len(a1) > 0:
        return ("admin1_name", "admin1_name IS NOT NULL")
    return ("admin0_name", "admin0_name IS NOT NULL")


def _query_hazard_by_crop(req: HazardByCropRequest) -> List[Dict[str, Any]]:
    """Aggregate exposure by hazard × crop for a single side.

    This endpoint must stay *snappy* for interactive charts, so we:
    - run exactly one parquet scan (GROUP BY hazard, crop)
    - apply top_hazards/top_crops pruning in Python on the already-aggregated result
      (result cardinality is small: hazards × crops)
    - optionally bucket non-top crops into "Other" (keeps totals consistent for stacked charts)
    """
    _validate_url(req.dataset_url)
    _parquet_magic_check(req.dataset_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection (admin0=all with no admin1/admin2) is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )

    geo_where_expr = _geo_where(req.geo)

    q = f"""
      SELECT hazard, crop, COALESCE(SUM(CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE) ELSE NULL END), 0.0) AS total
      FROM read_parquet({_sql_q(req.dataset_url)})
      WHERE {_scen_where(req.scen)}
        AND {geo_where_expr}
        AND {_crop_where(req.commodities)}
        AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
        AND {_haz_where(req.hazards)}
        AND hazard IS NOT NULL
        AND crop IS NOT NULL
      GROUP BY hazard, crop
    """

    con = _duckdb_connect(for_http_parquet=True)
    try:
        rows = _rows(con, q)
    finally:
        con.close()

    # Normalize totals to float (JSON-safe)
    for r in rows:
        try:
            t = float(r.get("total") or 0.0)
        except Exception:
            t = 0.0
        if not (t == t) or t in (float("inf"), float("-inf")):
            t = 0.0
        r["total"] = t

    # Top hazards (filter)
    if req.top_hazards and req.top_hazards > 0:
        haz_tot: Dict[str, float] = {}
        for r in rows:
            h = str(r.get("hazard") or "")
            haz_tot[h] = haz_tot.get(h, 0.0) + float(r["total"])
        keep = [h for h, _ in sorted(haz_tot.items(), key=lambda kv: kv[1], reverse=True)[: int(req.top_hazards)]]
        keep_set = set(keep)
        rows = [r for r in rows if str(r.get("hazard") or "") in keep_set]

    # Top crops: either bucket non-top into "Other" (default) or drop them (bucket_other=False).
    if req.top_crops and req.top_crops > 0:
        crop_tot: Dict[str, float] = {}
        for r in rows:
            c = str(r.get("crop") or "")
            crop_tot[c] = crop_tot.get(c, 0.0) + float(r["total"])
        keep = [c for c, _ in sorted(crop_tot.items(), key=lambda kv: kv[1], reverse=True)[: int(req.top_crops)]]
        keep_set = set(keep)

        if req.bucket_other is False:
            # Shiny-compatible: filter to top crops only (no "Other" bucket).
            rows = [r for r in rows if str(r.get("crop") or "") in keep_set]
        else:
            # Default: keep totals consistent for stacked charts by bucketing into "Other".
            agg: Dict[Tuple[str, str], float] = {}
            for r in rows:
                h = str(r.get("hazard") or "")
                c0 = str(r.get("crop") or "")
                c = c0 if c0 in keep_set else "Other"
                agg[(h, c)] = agg.get((h, c), 0.0) + float(r["total"])

            rows = [{"hazard": h, "crop": c, "total": t} for (h, c), t in agg.items()]

    # Sort hazards by their total, then crops within hazard by total
    haz_tot2: Dict[str, float] = {}
    for r in rows:
        h = str(r.get("hazard") or "")
        haz_tot2[h] = haz_tot2.get(h, 0.0) + float(r["total"])
    haz_order = {h: i for i, (h, _) in enumerate(sorted(haz_tot2.items(), key=lambda kv: kv[1], reverse=True))}

    rows.sort(
        key=lambda r: (
            haz_order.get(str(r.get("hazard") or ""), 10**9),
            -float(r.get("total") or 0.0),
            str(r.get("crop") or ""),
        )
    )

    return rows

def _query_by_admin(req: ByAdminRequest) -> List[Dict[str, Any]]:
    _validate_url(req.dataset_url)
    _parquet_magic_check(req.dataset_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )
    group_field, non_null = (_resolve_admin_group_fields(req.geo) if req.group_child else _resolve_admin_group_fields_current(req.geo))
    geo_where_expr = _geo_where_parent(req.geo) if req.group_child else _geo_where(req.geo)

    # Backward-compatible default: totals by admin only.
    # Optional Q4 mode: also group by hazard so the client can build compound/non-compound stacks.
    if bool(getattr(req, "group_hazard", False)):
        q = f"""
  SELECT
    {group_field} AS admin,
    CAST(hazard AS VARCHAR) AS hazard,
    COALESCE(
      SUM(
        CASE
          WHEN CAST(value AS DOUBLE) = CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE)
          ELSE NULL
        END
      ),
      0.0
    ) AS total
  FROM read_parquet({_sql_q(req.dataset_url)})
  WHERE {_scen_where(req.scen)}
    AND {geo_where_expr}
    AND {_crop_where(req.commodities)}
    AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
    AND {_haz_where(req.hazards)}
    AND {non_null}
    AND hazard IS NOT NULL
  GROUP BY admin, hazard
  ORDER BY admin ASC, total DESC
"""
    else:
        q = f"""
  SELECT
    {group_field} AS admin,
    COALESCE(
      SUM(
        CASE
          WHEN CAST(value AS DOUBLE) = CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE)
          ELSE NULL
        END
      ),
      0.0
    ) AS total
  FROM read_parquet({_sql_q(req.dataset_url)})
  WHERE {_scen_where(req.scen)}
    AND {geo_where_expr}
    AND {_crop_where(req.commodities)}
    AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
    AND {_haz_where(req.hazards)}
    AND {non_null}
  GROUP BY admin
  ORDER BY total DESC
"""

    con = _duckdb_connect(for_http_parquet=True)
    try:
        return _rows(con, q)
    finally:
        con.close()



def _denom_query_try(
    con: duckdb.DuckDBPyConnection,
    denom_url: str,
    base_wheres: List[str],
    exposure: Optional[str],
    unit: Optional[str],
    exposure_unit_legacy: Optional[str],
    group_by_crop: bool,
) -> Any:
    """Try denom queries against slightly different schemas.

    Supported denom schemas we see in Atlas/Shiny:
      A) legacy: columns include (admin0_name, admin1_name, admin2_name, crop, value, exposure_unit[, tech])
      B) harmonized: columns include (admin*_name, crop, value, exposure, unit[, tech, stat])

    We try combinations of possible column names to stay robust.
    """
    # column name variants
    exposure_cols = ["exposure", "exposure_short", "exposure_type"]
    unit_cols = ["unit", "exposure_unit"]

    # Resolve legacy single-field if provided
    exp = exposure
    unt = unit
    if exp is None and unt is None and exposure_unit_legacy:
        # If the value looks like a known exposure key, map to Shiny-like defaults.
        v = str(exposure_unit_legacy).strip()
        if v in ("prod", "area", "vop"):
            exp = {"prod": "prod", "area": "harv-area", "vop": "vop"}[v]
            unt = {"prod": "t", "area": "ha", "vop": "intld15"}[v]
        elif v in ("intld15", "usd", "usd15"):
            exp = "vop"
            unt = "usd" if v in ("usd", "usd15") else "intld15"
        elif v in ("people", "number"):
            exp, unt = "number", "number"
        else:
            # assume it's a unit (e.g., ha/t/usd/intld15)
            unt = v

    last_err: Optional[Exception] = None

    # Try: with exposure+unit; then unit only; then legacy exposure_unit only
    attempts = []
    if exp and unt:
        attempts.append(("exp+unit", exp, unt))
    if unt:
        attempts.append(("unit", None, unt))
    if exposure_unit_legacy:
        attempts.append(("legacy", None, exposure_unit_legacy))

    for _, exp_val, unit_val in attempts:
        for use_exp_col in ([True] if exp_val else [False]):
            for exp_col in (exposure_cols if use_exp_col else [None]):
                for unit_col in unit_cols:
                    for use_tech in (True, False):
                        wheres = list(base_wheres)

                        if exp_val and exp_col:
                            wheres.append(f"{exp_col} = {_sql_q(exp_val)}")
                        if unit_val:
                            wheres.append(f"{unit_col} = {_sql_q(unit_val)}")
                        if use_tech:
                            # Shiny uses: (tech='all' OR tech IS NULL)
                            wheres.append("(tech = 'all' OR tech IS NULL)")

                        where_sql = " AND ".join(wheres) if wheres else "TRUE"

                        if group_by_crop:
                            q = f"""
                              SELECT crop,
                                     COALESCE(SUM(CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE)
                                                       THEN CAST(value AS DOUBLE) ELSE NULL END), 0.0) AS denom
                              FROM read_parquet({_sql_q(denom_url)})
                              WHERE {where_sql}
                              GROUP BY crop
                            """
                        else:
                            q = f"""
                              SELECT COALESCE(SUM(CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE)
                                                       THEN CAST(value AS DOUBLE) ELSE NULL END), 0.0) AS denom
                              FROM read_parquet({_sql_q(denom_url)})
                              WHERE {where_sql}
                            """

                        try:
                            return _rows(con, q)
                        except Exception as e:
                            msg = str(e)
                            last_err = e
                            # schema mismatch: try next combo
                            if "Binder" in msg and ("Referenced column" in msg or "Column" in msg):
                                continue
                            # other errors should surface
                            raise

    if last_err:
        raise last_err
    return []



def _query_denom_total(req: DenomTotalRequest) -> Dict[str, Any]:
    _validate_url(req.denom_url)
    _parquet_magic_check(req.denom_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )

    wheres = [
        _geo_where(req.geo),
        _crop_where(req.commodities),
    ]

    con = _duckdb_connect(for_http_parquet=True)
    try:
        rows = _denom_query_try(
            con,
            req.denom_url,
            wheres,
            req.exposure,
            req.unit,
            req.exposure_unit,
            group_by_crop=False,
        )
        denom = rows[0].get("denom") if rows else None
        try:
            n = float(denom) if denom is not None else None
        except Exception:
            n = None

        ok = n is not None and n == n
        return {"ok": ok, "denom": n, "error": None if ok else "Denominator is missing/NaN"}
    finally:
        con.close()



def _query_denom_by_crop(req: DenomTotalRequest) -> List[Dict[str, Any]]:
    """Return total exposure grouped by crop from the denom parquet.

    This is used by Q2/Q4 to compute derived categories such as "no hazard" as:
      value_tot(crop) - value_any_hazard(crop)
    """
    _validate_url(req.denom_url)
    _parquet_magic_check(req.denom_url)

    if _is_broad_geo(req.geo) and not S.allow_broad_geo:
        raise HTTPException(
            status_code=400,
            detail="Broad geo selection is disabled. Select a specific admin0/admin1/admin2 or set ALLOW_BROAD_GEO=true.",
        )

    base_wheres = [
        _geo_where(req.geo),
        _crop_where(req.commodities),
    ]

    con = _duckdb_connect(for_http_parquet=True)
    try:
        rows = _denom_query_try(con, req.denom_url, base_wheres, req.exposure, req.unit, req.exposure_unit, group_by_crop=True)
    finally:
        con.close()

    # Normalize output keys to match older Shiny naming (value_tot)
    out: List[Dict[str, Any]] = []
    for r in rows:
        crop = r.get("crop")
        denom = r.get("denom")
        try:
            v = float(denom) if denom is not None else 0.0
        except Exception:
            v = 0.0
        out.append({"crop": crop, "value_tot": v})

    return out



def _export_records_csv(req: RecordsRequest) -> str:
    # Guardrail for CSV exports
    limit_rows = min(S.export_max_rows, max(1, int(req.page_size)))
    order = "value DESC" if req.sort == "value_desc" else "value ASC"

    geo_where_expr = _geo_where(req.geo)

    q = f"""
      SELECT admin0_name, admin1_name, admin2_name,
             scenario, timeframe, hazard, hazard_vars, crop,
             CASE WHEN CAST(value AS DOUBLE)=CAST(value AS DOUBLE) THEN CAST(value AS DOUBLE) ELSE NULL END AS value
      FROM read_parquet({_sql_q(req.dataset_url)})
      WHERE {_scen_where(req.scen)}
        AND {geo_where_expr}
        AND {_crop_where(req.commodities)}
        AND {_hazard_vars_where(req.hazard_vars, req.method, req.commodity_group)}
      ORDER BY {order}
      LIMIT {limit_rows}
    """

    con = _duckdb_connect(for_http_parquet=True)
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        # Use a double-quote character as the CSV quote char.
        con.execute(f"COPY ({q}) TO {_sql_q(tmp_path)} (HEADER TRUE, DELIMITER ',', QUOTE '\"')")
        return tmp_path
    finally:
        con.close()


def _cleanup_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


# ----------------------------
# Startup / shutdown
# ----------------------------

@app.on_event("startup")
async def startup() -> None:
    global redis, cache_store

    redis = Redis.from_url(S.redis_url, decode_responses=True)
    try:
        await redis.ping()
    except Exception as e:
        raise RuntimeError(f"Redis not reachable at {S.redis_url}: {e}")

    cache_store = CacheStore(redis)
    cache_store.init_materialized_cache()

    # Warm up DuckDB + httpfs
    con = _duckdb_connect(for_http_parquet=True)
    con.close()


@app.on_event("shutdown")
async def shutdown() -> None:
    global redis
    if redis is not None:
        await redis.close()


@app.get("/health")
async def health() -> Dict[str, Any]:
    r_ok = False
    try:
        if redis is not None:
            await redis.ping()
            r_ok = True
    except Exception:
        r_ok = False

    return {
        "ok": r_ok,
        "redis": r_ok,
        "duckdb_path": S.duckdb_path,
        "cache_materialize": S.cache_materialize,
        "allowed_hosts": ["*"] if S.allow_any_url else S.allowed_parquet_hosts,
        "allow_broad_geo": S.allow_broad_geo,
    }


# ----------------------------
# API endpoints
# ----------------------------

class CacheClearRequest(BaseModel):
    prefixes: Optional[List[str]] = Field(default=None, description="Key prefixes to clear (e.g., ['by_admin','q1']).")
    all: bool = Field(default=False, description="If true, clear all known hazard-exposure cache prefixes.")
    dry_run: bool = Field(default=False, description="If true, only count keys that would be deleted.")

# Restrict what can be cleared (avoids nuking unrelated Redis keys)
HZ_CACHE_PREFIXES: List[str] = [
    "totals_by_hazard",
    "totals_by_crop",
    "hazard_by_crop",
    "by_admin",
    "q1",
    "records",
    "denom_total",
]

def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None

async def _require_cache_admin(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
) -> None:
    """Admin guard for cache-clear endpoints.

    - Requires CACHE_CLEAR_TOKEN to be set on the server.
    - By default, only allows calls coming from localhost unless CACHE_CLEAR_LOCAL_ONLY=false.
      (If you're behind Nginx, you can keep it local-only and call via SSH port-forward.)
    """
    token = (S.cache_clear_token or "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="Cache clear is disabled (CACHE_CLEAR_TOKEN not set).")

    if S.cache_clear_local_only:
        host = (request.client.host if request.client else "")
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(status_code=403, detail="Cache clear is restricted to localhost.")

    provided = _extract_bearer(authorization) or (x_admin_token or "").strip()
    if provided != token:
        raise HTTPException(status_code=401, detail="Unauthorized.")

@app.get("/api/v1/hz/cache/prefixes")
async def cache_prefixes(_: Any = Depends(_require_cache_admin)) -> Dict[str, Any]:
    return {"ok": True, "prefixes": HZ_CACHE_PREFIXES}

@app.post("/api/v1/hz/cache/clear")
async def cache_clear(req: CacheClearRequest, _: Any = Depends(_require_cache_admin)) -> Dict[str, Any]:
    assert cache_store is not None
    prefixes = []
    if req.all:
        prefixes = list(HZ_CACHE_PREFIXES)
    elif req.prefixes:
        prefixes = [p for p in req.prefixes if p]

    if not prefixes:
        raise HTTPException(status_code=400, detail="Provide prefixes or set all=true.")

    # Safety: only allow known prefixes
    unknown = [p for p in prefixes if p not in HZ_CACHE_PREFIXES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown prefixes: {unknown}. Allowed: {HZ_CACHE_PREFIXES}")

    info = await cache_store.clear_prefixes(prefixes, dry_run=req.dry_run)
    return {"ok": True, "prefixes": prefixes, **info}

@app.post("/api/v1/hz/totals-by-hazard")
@app.post("/api/v1/hz/totals_by_hazard")
async def totals_by_hazard(req: TotalsByHazardRequest) -> Dict[str, Any]:
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("totals_by_hazard", payload)
    ttl = _ttl(req.cache_ttl_seconds)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, "data": cached}

    t0 = time.time()
    data = _query_totals_by_hazard(req)
    dt_ms = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, data, ttl_seconds=ttl)
    return {"ok": True, "cached": False, "t_ms": dt_ms, "data": data}


@app.post("/api/v1/hz/totals-by-crop")
@app.post("/api/v1/hz/totals_by_crop")
async def totals_by_crop(req: TotalsByCropRequest) -> Dict[str, Any]:
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("totals_by_crop", payload)
    ttl = _ttl(req.cache_ttl_seconds)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, "data": cached}

    t0 = time.time()
    data = _query_totals_by_crop(req)
    dt_ms = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, data, ttl_seconds=ttl)
    return {"ok": True, "cached": False, "t_ms": dt_ms, "data": data}



@app.post("/api/v1/hz/hazard-by-crop")
@app.post("/api/v1/hz/hazard_by_crop")
async def hazard_by_crop(req: HazardByCropRequest) -> Dict[str, Any]:
    """Return exposure aggregated by hazard × crop (single side)."""
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("hazard_by_crop", payload)
    ttl = _ttl(req.cache_ttl_seconds)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, "data": cached}

    t0 = time.time()
    data = _query_hazard_by_crop(req)
    dt_ms = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, data, ttl_seconds=ttl)
    return {"ok": True, "cached": False, "t_ms": dt_ms, "data": data}

@app.post("/api/v1/hz/by-admin")
@app.post("/api/v1/hz/by_admin")
async def by_admin(req: ByAdminRequest) -> Dict[str, Any]:
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("by_admin", payload)
    ttl = _ttl(req.cache_ttl_seconds)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, "data": cached}

    t0 = time.time()
    data = _query_by_admin(req)
    dt_ms = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, data, ttl_seconds=ttl)
    return {"ok": True, "cached": False, "t_ms": dt_ms, "data": data}


@app.post("/api/v1/exposure/denom-total")
@app.post("/api/v1/exposure/denom_total")
async def denom_total(req: DenomTotalRequest) -> Dict[str, Any]:
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("denom_total", payload)
    ttl = _ttl(req.cache_ttl_seconds)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"cached": True, "cache_source": source, **cached}

    t0 = time.time()
    data = _query_denom_total(req)
    dt_ms = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, data, ttl_seconds=ttl)
    return {"cached": False, "t_ms": dt_ms, **data}


@app.post("/api/v1/hz/q1")
async def q1(req: Q1Request) -> Dict[str, Any]:
    """Convenience endpoint for the Q1 chart.

    Returns left totals, right totals, merged diff rows, plus a denom (if supplied).
    """
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("q1", payload)
    # Use standard TTL semantics (0 => no expiry)
    ttl = _ttl(None)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, **cached}

    t0 = time.time()

    left_rows = _query_totals_by_hazard(req.left)
    right_rows = _query_totals_by_hazard(req.right)

    denom_meta = {"ok": False, "denom": None, "error": "No denom"}
    denom_value = None
    if req.denom is not None:
        denom_meta = _query_denom_total(req.denom)
        if denom_meta.get("ok"):
            denom_value = denom_meta.get("denom")

    by1 = {r.get("hazard"): float(r.get("total") or 0) for r in left_rows}
    by2 = {r.get("hazard"): float(r.get("total") or 0) for r in right_rows}

    hazards = sorted(set(list(by1.keys()) + list(by2.keys())))
    sum1 = sum(by1.values())
    sum2 = sum(by2.values())

    def pct(val: float, s: float) -> float:
        if denom_value is not None and denom_value and denom_value > 0:
            return (val / denom_value) * 100.0
        if s and s > 0:
            return (val / s) * 100.0
        return 0.0

    merged: List[Dict[str, Any]] = []
    for h in hazards:
        t1 = by1.get(h, 0.0)
        t2 = by2.get(h, 0.0)
        merged.append(
            {
                "hazard": h,
                "total1": t1,
                "total2": t2,
                "total_diff": t2 - t1,
                "perc1": pct(t1, sum1),
                "perc2": pct(t2, sum2),
                "pct_diff": pct(t2, sum2) - pct(t1, sum1),
            }
        )

    merged.sort(key=lambda r: abs(r.get("total_diff", 0.0)), reverse=True)

    dt_ms = int((time.time() - t0) * 1000)

    out = {
        "left": left_rows,
        "right": right_rows,
        "merged": merged,
        "denom": denom_meta,
        "relative_label": "% of total exposure" if denom_meta.get("ok") else "% of hazard sum (fallback)",
        "t_ms": dt_ms,
    }

    await cache_store.set_json(key, out, ttl_seconds=ttl)
    return {"ok": True, "cached": False, **out}



@app.post("/api/v1/hz/q2")
async def q2(req: Q2Request) -> Dict[str, Any]:
    """Convenience endpoint for the Q2 chart (crop-centric exposure).

    Returns hazard×crop rows for left/right scenarios plus (optionally) denom totals:
    - denom: single global total
    - denom_by_crop: per-crop totals for computing "no hazard" and percentages
    """
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("q2", payload)
    ttl = _ttl(None)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, **cached}

    t0 = time.time()

    left_rows = _query_hazard_by_crop(req.left)
    right_rows = _query_hazard_by_crop(req.right)

    denom_meta = {"ok": False, "denom": None, "error": "No denom"}
    denom_by_crop: Optional[List[Dict[str, Any]]] = None
    if req.denom is not None:
        denom_meta = _query_denom_total(req.denom)
        denom_by_crop = _query_denom_by_crop(req.denom)

    dt_ms = int((time.time() - t0) * 1000)

    out = {
        "left": left_rows,
        "right": right_rows,
        "denom": denom_meta,
        "denom_by_crop": denom_by_crop,
        "relative_label": "% of total exposure" if denom_meta.get("ok") else "% of hazard sum (fallback)",
        "t_ms": dt_ms,
    }

    await cache_store.set_json(key, out, ttl_seconds=ttl)
    return {"ok": True, "cached": False, **out}


@app.post("/api/v1/hz/records")
async def records(req: RecordsRequest) -> Dict[str, Any]:
    assert cache_store is not None

    payload = req.model_dump()
    key = _cache_key("records", payload)

    # Records pages are less reusable; cache briefly to make UI paging snappy.
    ttl = _ttl(req.cache_ttl_seconds)
    if isinstance(ttl, int) and ttl > 0:
        ttl = min(120, ttl)

    cached, source = await cache_store.get_json(key, ttl_seconds=ttl)
    if cached is not None:
        return {"ok": True, "cached": True, "cache_source": source, **cached}

    t0 = time.time()
    out = _query_records_page(req)
    out["t_ms"] = int((time.time() - t0) * 1000)

    await cache_store.set_json(key, out, ttl_seconds=ttl)
    return {"ok": True, "cached": False, **out}


@app.post("/api/v1/hz/records.csv")
@app.post("/api/v1/hz/records_csv")
async def records_csv(req: RecordsRequest, bg: BackgroundTasks) -> FileResponse:
    """Generate a CSV export server-side.

    Guardrail: EXPORT_MAX_ROWS (default 200k). Increase only if you really need it.
    """
    _validate_url(req.dataset_url)
    _parquet_magic_check(req.dataset_url)

    path = _export_records_csv(req)
    bg.add_task(_cleanup_file, path)

    filename = f"hazard_exposure_records_{int(time.time())}.csv"
    return FileResponse(path, filename=filename, media_type="text/csv")