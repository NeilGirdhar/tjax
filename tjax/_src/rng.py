from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr

from .annotations import JaxIntegralArray, KeyArray


class RngStream:
    """A stateful wrapper around a JAX PRNG key that produces an infinite sequence of keys.

    Each call to :meth:`key` folds the current counter into the base key and increments the
    counter, so successive calls yield independent, reproducible keys without requiring the
    caller to thread the key through their code.
    """

    def __init__(self, key: KeyArray, count: JaxIntegralArray | None = None) -> None:
        super().__init__()
        if count is None:
            count = jnp.zeros((), dtype=jnp.uint32)
        else:
            assert count.ndim == 0
        self._key = key
        self._count = count

    def key(self) -> KeyArray:
        """Return the next key in the stream and advance the internal counter."""
        key = jr.fold_in(self._key, self._count)
        self._count += 1
        return key

    def fork(self, samples: int) -> KeyArray:
        """Split the next key into ``samples`` independent sub-keys."""
        return jr.split(self.key(), samples)


def create_streams(keys: Mapping[str, KeyArray]) -> Mapping[str, RngStream]:
    """Wrap a mapping of named keys into a mapping of :class:`RngStream` objects."""
    return {name: RngStream(key) for name, key in keys.items()}


def sample_streams(rngs: Mapping[str, RngStream]) -> Mapping[str, KeyArray]:
    """Advance each stream by one step and return the resulting keys."""
    return {name: stream.key() for name, stream in rngs.items()}


def fork_streams(rngs: Mapping[str, RngStream], samples: int) -> Mapping[str, KeyArray]:
    """Fork each stream into ``samples`` independent sub-keys."""
    return {name: stream.fork(samples) for name, stream in rngs.items()}
