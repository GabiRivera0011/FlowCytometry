from __future__ import annotations

import hashlib
from pathlib import Path


def content_hash(file_or_path) -> str:
    """MD5 of the file's bytes. For a path, hashes by streaming in chunks so
    the whole file is never held in memory just to compute a cache key — for
    an uploaded file object, it's already fully buffered by the caller, so
    hashing its existing bytes adds no extra copy."""
    if isinstance(file_or_path, (str, Path)):
        hasher = hashlib.md5()
        with open(file_or_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    return hashlib.md5(file_or_path.getvalue()).hexdigest()
