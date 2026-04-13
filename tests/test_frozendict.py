from __future__ import annotations

import jax
import jax.numpy as jnp
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


def test_frozendict_is_registered_as_pytree() -> None:
    data = frozendict({"b": 2, "a": 1})

    leaves, treedef = jax.tree_util.tree_flatten(data)

    assert leaves == [1, 2]

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, frozendict)
    assert rebuilt == frozendict({"a": 1, "b": 2})


def test_frozendict_pytree_paths_use_dict_keys() -> None:
    data = frozendict({"b": 2, "a": 1})

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(data)

    assert [value for _, value in leaves_with_paths] == [1, 2]
    assert [path[0].key for path, _ in leaves_with_paths] == ["a", "b"]


def test_frozendict_can_flow_through_jit() -> None:
    @jax.jit
    def increment_leaves(data: frozendict[str, jax.Array]) -> frozendict[str, jax.Array]:
        return jax.tree_util.tree_map(lambda x: x + 1, data)

    result = increment_leaves(frozendict({"b": jnp.array(2), "a": jnp.array(1)}))

    assert isinstance(result, frozendict)
    assert list(result) == ["a", "b"]
    assert int(result["a"]) == 2  # noqa: PLR2004
    assert int(result["b"]) == 3  # noqa: PLR2004
