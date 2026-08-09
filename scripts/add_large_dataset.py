"""Utility script to download a large dataset (2‑4 GB) and register it in the AFE catalog.

Usage (from the repository root)::

    python scripts/add_large_dataset.py \
        --url https://example.com/path/to/dataset.tar.gz \
        --name "My Large Corpus" \
        --project-id <project-uuid> \
        [--description "Optional description"]

The script:
1. Streams the file to the local storage backend (via ``core.storage``).
2. Computes size and a simple SHA‑256 checksum.
3. Creates a ``Dataset`` DB record with ``status='raw'``.
4. Logs progress to the console.

It is intended for one‑off ingestion of public corpora (The Pile, Common Crawl
splits, Wikipedia dumps, etc.).  For production use you would likely replace
this with a more robust ingestion pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import requests

from core.storage import get_storage_backend
from core.database.engine import get_session
from core.database.models.dataset import Dataset
from core.database.models.project import Project
from core.tenancy import TenantService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_to_temp(url: str, chunk_size: int = 4 * 1024 * 1024) -> tuple[Path, int, str]:
    """Download a remote file to a temporary location.

    Returns the temporary file path, total bytes downloaded, and the SHA‑256
    checksum of the content.
    """
    hasher = hashlib.sha256()
    total = 0
    tmp_file = Path(tempfile.mkstemp(suffix=Path(url).suffix)[1])
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp_file, "wb") as out:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    out.write(chunk)
                    total += len(chunk)
                    hasher.update(chunk)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    return tmp_file, total, hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and register a large dataset.")
    parser.add_argument("--url", required=True, help="Direct URL to the dataset file (must support HTTP range).")
    parser.add_argument("--name", required=True, help="Human‑readable name for the dataset.")
    parser.add_argument("--project-id", required=True, help="UUID of the project to associate the dataset with.")
    parser.add_argument("--description", default="", help="Optional description.")
    args = parser.parse_args()

    storage = get_storage_backend()
    dataset_id = "ds-" + hashlib.sha256((args.url + args.name).encode()).hexdigest()[:12]
    storage_key = f"datasets/{dataset_id}/raw.bin"

    logger.info("Starting download of %s", args.url)
    try:
        tmp_path, total_bytes, sha256 = download_to_temp(args.url)
        # Upload the file to the storage backend using a file‑like object to avoid
        # loading the entire dataset into memory (important for 2‑4 GB files).
        with open(tmp_path, "rb") as f:
            storage.write(storage_key, f)
    except Exception as exc:
        logger.error("Failed to ingest dataset: %s", exc)
        sys.exit(1)
    finally:
        # Clean up temporary file if it exists.
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()

    logger.info("Download complete – %d bytes, sha256=%s", total_bytes, sha256)

    # Register in DB
    with get_session() as session:
        project = session.get(Project, args.project_id)
        if not project:
            logger.error("Project %s not found", args.project_id)
            sys.exit(1)
        ds = Dataset(
            id=dataset_id,
            name=args.name,
            slug=args.name.lower().replace(' ', '-')[:48] + f"-{dataset_id[:8]}",
            description=args.description,
            format=Path(args.url).suffix.lstrip('.'),
            storage_key=storage_key,
            size_bytes=total_bytes,
            row_count=0,  # unknown until ingestion step
            token_count=0,
            quality_score=0.0,
            status="raw",
            project_id=args.project_id,
            created_by=TenantService.current_user_id() or "system",
        )
        session.add(ds)
        session.commit()
        logger.info("Dataset %s registered with id %s", args.name, dataset_id)


if __name__ == "__main__":
    main()
