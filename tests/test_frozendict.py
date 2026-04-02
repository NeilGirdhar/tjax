from __future__ import annotations

import pytest

from tjax import frozendict


def test_frozendict_mapping_and_hash() -> None:
    data = frozendict({"a": 1, "b": 2})

    assert data["a"] == 1
    assert len(data) == 2  # noqa: PLR2004
    assert list(data) == ["a", "b"]
    assert hash(data) == hash(frozendict({"b": 2, "a": 1}))
    assert repr(data) == "frozendict({'a': 1, 'b': 2})"


def test_frozendict_is_immutable() -> None:
    data = frozendict({"a": 1})

    with pytest.raises(AttributeError, match="frozendict is immutable"):
        data.some_attr = 2

    with pytest.raises(AttributeError, match="frozendict is immutable"):
        del data.some_attr  # ty: ignore
