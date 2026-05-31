from pathlib import Path

from pydantic import BaseModel

from gpu_forecasters.cache import FileCache


class Entry(BaseModel, frozen=True):
    name: str
    value: int


def test_put_then_get_roundtrips(tmp_path: Path) -> None:
    cache: FileCache[Entry] = FileCache(root=tmp_path, value_type=Entry)
    cache.put("alpha", Entry(name="alpha", value=1))
    assert cache.get("alpha") == Entry(name="alpha", value=1)


def test_get_missing_returns_none(tmp_path: Path) -> None:
    cache: FileCache[Entry] = FileCache(root=tmp_path, value_type=Entry)
    assert cache.get("nope") is None


def test_slash_key_creates_subdirectory(tmp_path: Path) -> None:
    cache: FileCache[Entry] = FileCache(root=tmp_path, value_type=Entry)
    cache.put("bucket/alpha", Entry(name="alpha", value=2))
    assert (tmp_path / "bucket" / "alpha.json").exists()
    assert cache.get("bucket/alpha") == Entry(name="alpha", value=2)


def test_put_is_overwrite(tmp_path: Path) -> None:
    cache: FileCache[Entry] = FileCache(root=tmp_path, value_type=Entry)
    cache.put("k", Entry(name="k", value=1))
    cache.put("k", Entry(name="k", value=2))
    assert cache.get("k") == Entry(name="k", value=2)
