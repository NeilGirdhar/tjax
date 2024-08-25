from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from jax.random import fold_in, split

from .annotations import JaxIntegralArray, KeyArray


class RngStream:
    def __init__(self, key: KeyArray, count: JaxIntegralArray | None = None):
        super().__init__()
        if count is None:
            count = jnp.zeros((), dtype=jnp.uint32)
        else:
            assert count.ndim == 0
        self._key = key
        self.count = count

    def key(self) -> KeyArray:
        key = fold_in(self._key, self.count)
        self.count += 1
        return key


def fork_streams(rngs: Mapping[str, RngStream],
                 samples: int | None = None
                 ) -> Mapping[str, KeyArray]:
    if samples is None:
        return {name: stream.key() for name, stream in rngs.items()}
    return {name: split(stream.key(), samples) for name, stream in rngs.items()}


def create_streams(keys: Mapping[str, KeyArray]) -> Mapping[str, RngStream]:
    return {name: RngStream(key) for name, key in keys.items()}
