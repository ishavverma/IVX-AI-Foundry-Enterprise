"""Download bounded samples from the approved dataset catalog into IVX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import get_session, init_db
from core.database.models.dataset import Dataset
from core.storage.paths import storage_paths
from core.storage.sdk import get_storage_backend

CATALOG_PATH = BASE_DIR / "data" / "preloaded_datasets.json"
VIEWER_API = "https://datasets-server.huggingface.co/rows"


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def fetch_rows(entry: dict, limit: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page_size = min(100, limit)

    while len(rows) < limit:
        query = urlencode({
            "dataset": entry["dataset"],
            "config": entry["config"],
            "split": entry["split"],
            "offset": offset,
            "length": page_size,
        })
        request = Request(VIEWER_API + "?" + query, headers={"User-Agent": "AFE-dataset-preloader/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)

        page = payload.get("rows", [])
        if not page:
            break
        rows.extend(item.get("row", {}) for item in page)
        offset += len(page)
        if len(page) < page_size:
            break

    return rows[:limit]


def row_to_text(row: dict) -> str:
    if isinstance(row.get("text"), str):
        return row["text"].strip()

    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                role = message.get("role", "user")
                parts.append(f"{role}: {message['content']}")
        return "\n".join(parts).strip()

    for key in ("prompt", "completion", "response", "content"):
        if row.get(key):
            return str(row[key]).strip()
    return " ".join(str(value) for value in row.values() if value).strip()


def preload(name: str, limit: int, overwrite: bool = False) -> dict:
    catalog = load_catalog()
    if name not in catalog:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(sorted(catalog))}")

    entry = catalog[name]
    dataset_id = str(uuid.uuid5(uuid.NAMESPACE_URL, entry["source_url"]))
    storage_key = f"datasets/{dataset_id}/preloaded.jsonl"

    init_db()
    storage_paths.init_all()
    storage = get_storage_backend()
    with get_session() as session:
        existing = session.get(Dataset, dataset_id)
        if existing and not overwrite:
            print(f"Already loaded: {existing.name} ({existing.row_count} rows). Use --overwrite to refresh.")
            return {"id": dataset_id, "status": "existing", "rows": existing.row_count}

    rows = fetch_rows(entry, limit)
    records = [row for row in (row_to_text(row) for row in rows) if row]
    if not records:
        raise RuntimeError(f"No usable rows returned for {entry['dataset']}")

    content = "\n".join(json.dumps({"text": text}, ensure_ascii=False) for text in records).encode("utf-8")
    storage.write(storage_key, content)
    digest = hashlib.sha256(content).hexdigest()
    local_path = storage.get_url(storage_key)

    from engines.dataset.ingestion import DatasetIngestion
    samples, row_count, size_bytes = DatasetIngestion.sample_head(local_path, n=5)
    lineage = {
        "source": entry,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_rows": limit,
        "source_rows_downloaded": len(rows),
        "sha256": digest,
        "preload_mode": "huggingface_dataset_viewer_sample",
    }

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            dataset = Dataset(id=dataset_id)
            session.add(dataset)
        dataset.name = entry["name"]
        dataset.slug = f"preloaded-{name}"
        dataset.description = entry["description"]
        dataset.format = "jsonl"
        dataset.storage_key = storage_key
        dataset.size_bytes = size_bytes
        dataset.row_count = row_count
        dataset.status = "ready"
        dataset.sample_preview_json = json.dumps(samples, ensure_ascii=False)
        dataset.lineage_json = json.dumps(lineage, ensure_ascii=False)

    return {"id": dataset_id, "status": "loaded", "rows": row_count, "bytes": size_bytes, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(load_catalog()))
    parser.add_argument("--limit", type=int, default=1000, help="Maximum rows to download (default: 1000)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing preloaded dataset")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100_000:
        parser.error("--limit must be between 1 and 100000")
    print(json.dumps(preload(args.dataset, args.limit, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
