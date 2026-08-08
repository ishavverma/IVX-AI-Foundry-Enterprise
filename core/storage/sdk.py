"""
AFE — Storage SDK

A unified abstraction over storage backends.  Application code never
touches the filesystem directly — it always goes through this SDK so
the underlying backend (local disk today, S3/MinIO tomorrow) can be
swapped without changing any calling code.

Usage:
    from core.storage import get_storage_backend

    storage = get_storage_backend()
    storage.write("datasets/my_dataset/data.jsonl", raw_bytes)
    content = storage.read("datasets/my_dataset/data.jsonl")
    files   = storage.list_dir("datasets/my_dataset")
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, List, Optional

from core.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


# ──────────────────────────────────────────────
# Abstract Backend Interface
# ──────────────────────────────────────────────

class StorageBackend(ABC):
    """Interface that every storage driver must implement."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read the entire file at *key* and return its bytes."""

    @abstractmethod
    def read_stream(self, key: str) -> BinaryIO:
        """Return a file-like stream for the file at *key*."""

    @abstractmethod
    def write(self, key: str, data: bytes | BinaryIO) -> str:
        """Write *data* to *key*.  Return the canonical key."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete the file at *key*.  Return True if deleted."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists."""

    @abstractmethod
    def list_dir(self, prefix: str) -> List[str]:
        """List all keys under *prefix*."""

    @abstractmethod
    def size(self, key: str) -> int:
        """Return file size in bytes."""

    @abstractmethod
    def copy(self, src: str, dst: str) -> str:
        """Copy *src* → *dst*.  Return the destination key."""

    @abstractmethod
    def move(self, src: str, dst: str) -> str:
        """Move *src* → *dst*.  Return the destination key."""

    @abstractmethod
    def get_url(self, key: str) -> str:
        """Return a URL / path suitable for serving the file."""


# ──────────────────────────────────────────────
# Local Filesystem Driver
# ──────────────────────────────────────────────

class LocalStorageBackend(StorageBackend):
    """Stores files on the local filesystem under STORAGE_ROOT."""

    def __init__(self, root: str | None = None) -> None:
        self._root = Path(root or _settings.STORAGE_ROOT)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorage initialised at %s", self._root)

    def _resolve(self, key: str) -> Path:
        """Resolve a key to an absolute path, preventing directory traversal."""
        root = self._root.resolve()
        resolved = (root / key).resolve()
        if not resolved.is_relative_to(root):
            raise PermissionError(f"Path traversal detected: {key}")
        return resolved

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return path.read_bytes()

    def read_stream(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return open(path, "rb")  # noqa: SIM115

    def write(self, key: str, data: bytes | BinaryIO) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            with open(path, "wb") as f:
                shutil.copyfileobj(data, f)
        logger.debug("Wrote %s (%d bytes)", key, path.stat().st_size)
        return key

    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if path.is_file():
            path.unlink()
            logger.debug("Deleted %s", key)
            return True
        if path.is_dir():
            shutil.rmtree(path)
            logger.debug("Deleted directory %s", key)
            return True
        return False

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list_dir(self, prefix: str) -> List[str]:
        path = self._resolve(prefix)
        if not path.is_dir():
            return []
        root_str = str(self._root.resolve())
        results = []
        for item in sorted(path.rglob("*")):
            if item.is_file():
                rel = str(item.resolve()).replace(root_str, "").lstrip(os.sep)
                results.append(rel.replace(os.sep, "/"))
        return results

    def size(self, key: str) -> int:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return path.stat().st_size

    def copy(self, src: str, dst: str) -> str:
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        return dst

    def move(self, src: str, dst: str) -> str:
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        return dst

    def get_url(self, key: str) -> str:
        return str(self._resolve(key))

    @property
    def root(self) -> Path:
        return self._root


# ──────────────────────────────────────────────
# S3 / MinIO Driver (stub — activates when S3 is configured)
# ──────────────────────────────────────────────

class S3StorageBackend(StorageBackend):
    """
    S3-compatible storage driver.

    Requires the `boto3` package.  Configured via environment variables:
        AFE_S3_ENDPOINT, AFE_S3_ACCESS_KEY, AFE_S3_SECRET_KEY,
        AFE_S3_BUCKET, AFE_S3_REGION
    """

    def __init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install it: pip install boto3")

        kwargs = {
            "aws_access_key_id": _settings.S3_ACCESS_KEY,
            "aws_secret_access_key": _settings.S3_SECRET_KEY,
            "region_name": _settings.S3_REGION,
            "config": BotoConfig(signature_version="s3v4"),
        }
        if _settings.S3_ENDPOINT:
            kwargs["endpoint_url"] = _settings.S3_ENDPOINT

        self._s3 = boto3.client("s3", **kwargs)
        self._bucket = _settings.S3_BUCKET

        # Ensure bucket exists
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception:
            self._s3.create_bucket(Bucket=self._bucket)
            logger.info("Created S3 bucket: %s", self._bucket)

        logger.info("S3Storage initialised (bucket=%s, endpoint=%s)", self._bucket, _settings.S3_ENDPOINT or "AWS")

    def read(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    def read_stream(self, key: str) -> BinaryIO:
        resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"]

    def write(self, key: str, data: bytes | BinaryIO) -> str:
        if isinstance(data, bytes):
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
        else:
            self._s3.upload_fileobj(data, self._bucket, key)
        return key

    def delete(self, key: str) -> bool:
        self._s3.delete_object(Bucket=self._bucket, Key=key)
        return True

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def list_dir(self, prefix: str) -> List[str]:
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def size(self, key: str) -> int:
        resp = self._s3.head_object(Bucket=self._bucket, Key=key)
        return resp["ContentLength"]

    def copy(self, src: str, dst: str) -> str:
        self._s3.copy_object(
            Bucket=self._bucket,
            CopySource={"Bucket": self._bucket, "Key": src},
            Key=dst,
        )
        return dst

    def move(self, src: str, dst: str) -> str:
        self.copy(src, dst)
        self.delete(src)
        return dst

    def get_url(self, key: str) -> str:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=3600,
        )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

_backend_instance: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend singleton."""
    global _backend_instance
    if _backend_instance is None:
        if _settings.STORAGE_BACKEND == "s3":
            _backend_instance = S3StorageBackend()
        else:
            _backend_instance = LocalStorageBackend()
    return _backend_instance
