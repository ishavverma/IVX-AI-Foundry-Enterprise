"""Dataset source connectors for public and authenticated repositories."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class DatasetSource:
    """A resolved source ready to be downloaded into managed storage."""

    source_type: str
    uri: str
    filename: str
    headers: dict[str, str] | None = None


class DatasetConnector:
    """Resolve and download dataset sources without bypassing the Storage SDK."""

    timeout_seconds = 60

    @staticmethod
    def github(owner: str, repo: str, path: str, ref: str = "main") -> DatasetSource:
        filename = Path(path).name or "dataset"
        uri = f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/{quote(ref)}/{quote(path)}"
        return DatasetSource("github", uri, filename)

    @staticmethod
    def huggingface(repo_id: str, filename: str, revision: str = "main") -> DatasetSource:
        uri = f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}/resolve/{quote(revision)}/{quote(filename, safe='/')}"
        token = os.getenv("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return DatasetSource("huggingface", uri, Path(filename).name, headers)

    @staticmethod
    def kaggle(dataset: str, filename: str) -> DatasetSource:
        uri = f"https://www.kaggle.com/api/v1/datasets/download/{quote(dataset, safe='/')}?filename={quote(filename)}"
        token = os.getenv("KAGGLE_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return DatasetSource("kaggle", uri, Path(filename).name, headers)

    @classmethod
    def http(cls, uri: str, filename: str | None = None) -> DatasetSource:
        return DatasetSource("http", uri, filename or Path(uri.split("?", 1)[0]).name or "dataset")

    @classmethod
    def download(cls, source: DatasetSource) -> BinaryIO:
        response = requests.get(source.uri, headers=source.headers, timeout=cls.timeout_seconds, stream=True)
        response.raise_for_status()
        content = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                content.write(chunk)
        content.seek(0)
        return content

    @classmethod
    def download_to_storage(cls, source: DatasetSource, storage: object, key_prefix: str) -> str:
        """Download a source and write it using a StorageBackend-compatible object."""
        key = f"{key_prefix.rstrip('/')}/{source.filename}"
        stream = cls.download(source)
        storage.write(key, stream)
        return key
