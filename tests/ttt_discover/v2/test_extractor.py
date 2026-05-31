from gpu_forecasters.ttt_discover.v2.extractors.python_block import (
    LastPythonBlockExtractor,
)


def test_returns_last_python_block() -> None:
    text = "draft:\n```python\nx = 1\n```\nfinal:\n```python\ny = 2\n```"
    assert LastPythonBlockExtractor().extract(text) == "y = 2"


def test_none_when_no_block() -> None:
    assert LastPythonBlockExtractor().extract("no blocks here") is None


def test_none_when_empty() -> None:
    assert LastPythonBlockExtractor().extract("") is None


def test_ignores_cpp_blocks() -> None:
    text = "```cpp\nint x = 0;\n```\nand:\n```python\nreturn 1\n```"
    assert LastPythonBlockExtractor().extract(text) == "return 1"


def test_none_when_only_cpp_blocks() -> None:
    text = "```cpp\nint x = 0;\n```"
    assert LastPythonBlockExtractor().extract(text) is None


def test_empty_python_block_is_none() -> None:
    assert LastPythonBlockExtractor().extract("```python\n```") is None
