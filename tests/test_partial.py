from __future__ import annotations

import jax.numpy as jnp
from jax import tree

from tjax import Partial


def f(a: object, b: object, c: object, *, scale: object, offset: object) -> tuple[object, ...]:
    return a, b, c, scale, offset


def test_partial_round_trip() -> None:
    partial = Partial(
        f,
        "static",
        jnp.array([1.0, 2.0]),
        3,
        callable_is_static=True,
        static_argnums=(0, 2),
        static_kwargs={"scale": 5},
        offset=jnp.array([4.0, 5.0]),
    )

    leaves, tree_def = tree.flatten(partial)
    rebuilt = tree.unflatten(tree_def, leaves)

    assert rebuilt() == partial()


def test_partial_dynamic_callable_round_trip() -> None:
    partial = Partial(
        f,
        jnp.array([1.0]),
        callable_is_static=False,
        static_argnums=(),
        offset=jnp.array([2.0]),
    )

    leaves, tree_def = tree.flatten(partial)
    rebuilt = tree.unflatten(tree_def, leaves)

    assert rebuilt("arg", 3, scale=4) == partial("arg", 3, scale=4)
