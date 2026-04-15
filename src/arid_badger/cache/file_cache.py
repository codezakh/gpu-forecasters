"""Disk-backed cache of pydantic values — one JSON file per entry.

Symmetric `Mapping[str, V]`-style interface: both `get` and `put` take an
opaque string key. The caller owns key derivation, so domain-specific key
schemes (content hashes, composite `problem/sha` paths, etc.) live as pure
functions at the call site rather than being baked into the cache.

An entry's on-disk location is `root / f"{key}.json"`. A key that contains
slashes, like `"L2_83/abcd1234"`, produces a subdirectory automatically —
that's the mechanism for namespacing entries.

Writes are atomic via `os.replace`, so a crash never leaves a partial file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, TypeAdapter

V = TypeVar("V", bound=BaseModel)


class FileCache(Generic[V]):
    _root: Path
    _adapter: TypeAdapter[V]

    def __init__(self, *, root: Path, value_type: type[V]) -> None:
        self._root = root
        self._adapter = TypeAdapter(value_type)

    def _path(self, key: str) -> Path:
        """Pure: returns the path where an entry with this key would live. No I/O."""
        return self._root / f"{key}.json"

    def get(self, key: str) -> V | None:
        path = self._path(key)
        if not path.exists():
            return None
        return self._adapter.validate_json(path.read_bytes())

    def put(self, key: str, value: V) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        _ = tmp.write_bytes(self._adapter.dump_json(value))
        os.replace(tmp, path)
