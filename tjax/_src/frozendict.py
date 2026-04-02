from __future__ import annotations

import sys

if sys.version_info >= (3, 15):
    from builtins import frozendict
else:
    from collections.abc import Iterator, Mapping

    class frozendict[K, V](Mapping[K, V]):  # noqa: N801
        """An immutable, hashable dictionary backport for Python < 3.15.

        On Python 3.15+ the builtin ``frozendict`` is used instead.  This
        implementation stores items in an ordinary ``dict`` and computes the hash
        lazily on first access.

        Mutation attempts via ``__setattr__`` or ``__delattr__`` raise
        ``AttributeError``.
        """

        __slots__ = ("_data", "_hash")

        _data: dict[K, V]
        _hash: int | None

        def __init__(self, *args: object, **kwargs: object) -> None:
            object.__setattr__(self, "_data", dict(*args, **kwargs))
            object.__setattr__(self, "_hash", None)

        def __getitem__(self, key: K) -> V:
            return self._data[key]

        def __iter__(self) -> Iterator[K]:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

        def __hash__(self) -> int:
            if self._hash is None:
                object.__setattr__(self, "_hash", hash(frozenset(self._data.items())))
            assert isinstance(self._hash, int)
            return self._hash

        def __repr__(self) -> str:
            return f"{type(self).__name__}({self._data!r})"

        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError("frozendict is immutable")  # noqa: TRY003

        def __delattr__(self, name: str) -> None:
            raise AttributeError("frozendict is immutable")  # noqa: TRY003
