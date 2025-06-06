from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr

from .annotations import JaxIntegralArray, KeyArray


class RngStream:
    def __init__(self, key: KeyArray, count: JaxIntegralArray | None = None) -> None:
        super().__init__()
        if count is None:
            count = jnp.zeros((), dtype=jnp.uint32)
        else:
            assert count.ndim == 0
        self._key = key
        self._count = count

    def key(self) -> KeyArray:
        key = jr.fold_in(self._key, self._count)
        self._count += 1
        return key

    def fork(self, samples: int) -> KeyArray:
        return jr.split(self.key(), samples)


def create_streams(keys: Mapping[str, KeyArray]) -> Mapping[str, RngStream]:
    return {name: RngStream(key) for name, key in keys.items()}


def sample_streams(rngs: Mapping[str, RngStream]) -> Mapping[str, KeyArray]:
    return {name: stream.key() for name, stream in rngs.items()}


def fork_streams(rngs: Mapping[str, RngStream], samples: int) -> Mapping[str, KeyArray]:
    return {name: stream.fork(samples) for name, stream in rngs.items()}
