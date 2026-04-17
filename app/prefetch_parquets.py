import argparse
import hashlib
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import httpx

# Inventory from nbData.json
NB_DATA_S3_URLS: List[str] = [
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=historic/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=historic/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=ENSEMBLE/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=ENSEMBLE/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=historic/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=historic/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=ENSEMBLE/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=ENSEMBLE/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=historic/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=historic/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=annual/model=ENSEMBLE/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_usd15/period=jagermeyr/model=ENSEMBLE/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=historic/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=historic/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=ENSEMBLE/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=ENSEMBLE/severity=moderate/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=historic/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=historic/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=ENSEMBLE/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=ENSEMBLE/severity=severe/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=historic/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=historic/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=annual/model=ENSEMBLE/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=hazard_exposure/source=atlas_cmip6/region=ssa/processing=hazard-risk-exposure/variable=vop_intld15/period=jagermeyr/model=ENSEMBLE/severity=extreme/interaction.parquet",
    "s3://digital-atlas/domain=exposure/type=combined/source=glw4-2020_spam2020AA/region=ssa/processing=atlas-harmonized/variable=crop-livestock_all.parquet",
]

# Denominator
QMD_HARDCODED_DENOM_URL = (
    "https://digital-atlas.s3.amazonaws.com/"
    "domain=exposure/type=combined/source=glw4-2020_spam2020AA/"
    "region=ssa/processing=atlas-harmonized/variable=vop_nominal-usd-2015.parquet"
)


def canonical_public_https_url(url: str) -> str:
    """Match the app's public-HTTPS canonicalization."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme == "https":
        return url.strip()

    if scheme == "s3":
        bucket = parsed.netloc.strip()
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid s3:// URL: {url}")

        if bucket == "digital-atlas":
            return f"https://digital-atlas.s3.amazonaws.com/{key}"

        return f"https://{bucket}.s3.amazonaws.com/{key}"

    raise ValueError(f"Unsupported URL scheme for {url!r}")


def safe_local_parquet_path(cache_dir: Path, url: str) -> Path:
    canonical = canonical_public_https_url(url)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    basename = os.path.basename(urlparse(canonical).path) or "data.parquet"
    return cache_dir / f"{digest}_{basename}"


def build_url_inventory() -> List[str]:
    urls = list(NB_DATA_S3_URLS)
    urls.append(QMD_HARDCODED_DENOM_URL)

    deduped: List[str] = []
    seen = set()
    for url in urls:
        canonical = canonical_public_https_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(url)
    return deduped


def verify_parquet_header(path: Path) -> None:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != b"PAR1":
        raise ValueError(f"{path} is not a parquet file (missing PAR1 header)")


print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def download_one(
    source_url: str,
    cache_dir: Path,
    timeout_seconds: int,
    force: bool,
    verify_only: bool,
) -> str:
    canonical = canonical_public_https_url(source_url)
    out_path = safe_local_parquet_path(cache_dir, source_url)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        if verify_only:
            verify_parquet_header(out_path)
        return f"EXISTS {out_path.name}"

    if verify_only:
        if not out_path.exists():
            return f"MISSING {out_path.name}"
        verify_parquet_header(out_path)
        return f"OK {out_path.name}"

    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".parquet",
        prefix="prefetch_",
        dir=str(cache_dir),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with httpx.stream("GET", canonical, timeout=timeout_seconds, follow_redirects=True) as r:
            r.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in r.iter_bytes():
                    if chunk:
                        f.write(chunk)

        verify_parquet_header(tmp_path)
        os.replace(tmp_path, out_path)
        return f"DOWNLOADED {out_path.name}"
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch public parquet files into the app cache.")
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("PARQUET_CACHE_DIR", "/data/parquet_cache"),
        help="Cache directory matching the API's PARQUET_CACHE_DIR",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2))),
        help="Number of concurrent downloads",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("PARQUET_DOWNLOAD_TIMEOUT_SECONDS", "600")),
        help="HTTP download timeout in seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target cache file already exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Do not download; only verify that expected cache files already exist and look like parquet",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    urls = build_url_inventory()

    log(f"Cache dir: {cache_dir}")
    log(f"Total unique files to process: {len(urls)}")

    ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_one,
                url,
                cache_dir,
                int(args.timeout),
                bool(args.force),
                bool(args.verify),
            ): url
            for url in urls
        }

        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                log(result)
                ok += 1
            except Exception as e:
                log(f"FAILED {url} :: {e}")
                failed += 1

    log(f"Finished. success={ok} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
