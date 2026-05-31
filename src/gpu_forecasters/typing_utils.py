"""
This module contains a type for representing either a success value of type T or an error of type E.
Similar to Result/Either types in other languages.

Usage:
```python
from arid_badger.typing_utils import Option, Result, Error

def make_option() -> Option[int, str]:
    if random.random() > 0.5:
        return Result(1)
    else:
        return Error("error")

maybe = make_option()
if maybe.is_ok():
    print(maybe.unwrap())
else:
    print(maybe.unwrap_err())
```
"""

from typing import Generic, TypeVar, NoReturn, Callable, Any, Type, TypeGuard
import functools
from pydantic import BaseModel


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


def identity(x: T) -> T:
    return x


class Option(Generic[T, E]):
    """
    A type that represents either a success value of type T or an error of type E.
    Similar to Result/Either types in other languages.
    """

    def __init__(self) -> None:
        raise TypeError("Can't instantiate Option directly. Use Ok() or Err()")

    @staticmethod
    def ok(value: T) -> "Ok[T, E]":
        return Ok(value)

    @staticmethod
    def err(error: E) -> "Err[T, E]":
        return Err(error)

    def is_ok(self, /) -> bool:
        raise NotImplementedError

    def is_err(self, /) -> bool:
        raise NotImplementedError

    def unwrap(self) -> T:
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T:
        raise NotImplementedError

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        raise NotImplementedError

    def map(self, fn: Callable[[T], U]) -> "Option[U, E]":
        raise NotImplementedError

    def unwrap_err(self) -> E:
        """
        Returns the contained Err value.

        Raises:
            ValueError: If the option is Ok

        Examples:
            >>> result = Option.err("not found")
            >>> result.unwrap_err()
            'not found'
            >>> Option.ok(42).unwrap_err()  # raises ValueError
        """
        raise NotImplementedError


class Ok(Option[T, E]):
    def __init__(self, value: T) -> None:
        self._value = value

    def is_ok(self, /) -> bool:
        return True

    def is_err(self, /) -> bool:
        return False

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self._value

    def map(self, fn: Callable[[T], U]) -> "Option[U, E]":
        return Ok(fn(self._value))

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    def unwrap_err(self) -> NoReturn:
        raise ValueError(f"Called unwrap_err on an Ok value: {self._value}")


class Err(Option[T, E]):
    def __init__(self, error: E) -> None:
        self._error = error

    def is_ok(self, /) -> bool:
        return False

    def is_err(self, /) -> bool:
        return True

    def unwrap(self) -> NoReturn:
        raise ValueError(f"Called unwrap on an Err value: {self._error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return fn()

    def map(self, fn: Callable[[T], U]) -> "Option[U, E]":
        return Err(self._error)

    def __repr__(self) -> str:
        return f"Err({self._error!r})"

    def unwrap_err(self) -> E:
        return self._error


def is_ok(option: Option[T, E], /) -> TypeGuard[Ok[T, E]]:
    return isinstance(option, Ok)


def is_err(option: Option[T, E], /) -> TypeGuard[Err[T, E]]:
    return isinstance(option, Err)


class Result(Ok[T, Any]):
    """
    Avoid having to specify Any for the error type.
    """

    def __init__(self, value: T) -> None:
        super().__init__(value)


class Error(Err[Any, E]):
    """
    Avoid having to specify Any for the error type.
    """

    def __init__(self, error: E) -> None:
        super().__init__(error)


P = TypeVar("P")
Q = TypeVar("Q")


def implements(protocol: Type[P]) -> Callable[[Type[P]], Type[P]]:
    """
    A decorator that ensures that a class implements a protocol.

    Usage:
    ```python
    from typing import Protocol
    class CatProtocol(Protocol):
        def meow(self) -> str:
            ...

    @implements(CatProtocol)
    class Cat:
        def meow(self) -> str:
            return "meow"

    implements(CatProtocol)(Cat)
    """

    def decorator(cls: Type[P]) -> Type[P]:
        # The type checker will enforce that `cls` matches the `protocol` without casting.
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        # Returning the original class, which must be type-compatible with the protocol
        return cls

    return decorator
