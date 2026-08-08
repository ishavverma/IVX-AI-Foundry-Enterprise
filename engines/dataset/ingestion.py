"""
AFE — Dataset Ingestion Engine

Handles parsing, schema inference, format detection, and ingestion of datasets
from multiple sources: CSV, JSON, JSONL, Parquet, and raw text files.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)


class DatasetIngestion:
    """Parses and ingests heterogeneous dataset files into standard text streams."""

    @staticmethod
    def detect_format(file_path: Path | str) -> str:
        """Inspect file extension and initial bytes to determine format."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".jsonl":
            return "jsonl"
        elif ext == ".json":
            return "json"
        elif ext == ".csv":
            return "csv"
        elif ext == ".parquet":
            return "parquet"
        elif ext in [".txt", ".md", ".log"]:
            return "txt"
        
        # Fallback inspection
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line.startswith("{") and first_line.endswith("}"):
                    return "jsonl"
                elif first_line.startswith("[") or first_line.startswith("{"):
                    return "json"
                elif "," in first_line:
                    return "csv"
        except Exception:
            pass
        return "txt"

    @classmethod
    def read_records(cls, file_path: Path | str, text_field: str = "text", max_rows: int | None = None) -> Iterator[Dict[str, Any]]:
        """Yield dictionary records with at least a 'text' key."""
        path = Path(file_path)
        fmt = cls.detect_format(path)
        count = 0

        if fmt == "jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            if "text" not in record and text_field in record:
                                record["text"] = str(record[text_field])
                            elif "text" not in record:
                                # take the longest string value
                                str_vals = [str(v) for v in record.values() if isinstance(v, (str, int, float))]
                                record["text"] = " ".join(str_vals)
                            yield record
                            count += 1
                            if max_rows and count >= max_rows:
                                break
                    except Exception:
                        continue

        elif fmt == "json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                if "text" not in item and text_field in item:
                                    item["text"] = str(item[text_field])
                                yield item
                            else:
                                yield {"text": str(item)}
                            count += 1
                            if max_rows and count >= max_rows:
                                break
                    elif isinstance(data, dict):
                        yield {"text": str(data.get(text_field, json.dumps(data)))}
                except Exception as e:
                    logger.warning("Error reading JSON file %s: %s", path, e)

        elif fmt == "csv":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" not in row and text_field in row:
                        row["text"] = row[text_field]
                    elif "text" not in row:
                        row["text"] = " ".join([str(v) for v in row.values() if v])
                    yield row
                    count += 1
                    if max_rows and count >= max_rows:
                        break

        else: # txt
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                buffer = []
                for line in f:
                    if not line.strip() and buffer:
                        yield {"text": "".join(buffer).strip()}
                        buffer = []
                        count += 1
                        if max_rows and count >= max_rows:
                            break
                    else:
                        buffer.append(line)
                if buffer and (not max_rows or count < max_rows):
                    yield {"text": "".join(buffer).strip()}

    @classmethod
    def sample_head(cls, file_path: Path | str, n: int = 5) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Return a tuple of:
        (first n records, total estimated lines/rows, file size bytes)
        """
        path = Path(file_path)
        size_bytes = path.stat().st_size if path.exists() else 0
        samples: List[Dict[str, Any]] = []
        total_rows = 0

        for record in cls.read_records(path):
            if len(samples) < n:
                samples.append(record)
            total_rows += 1

        return samples, total_rows, size_bytes
