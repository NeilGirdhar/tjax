from __future__ import annotations

from typing import Self, TypeVar, override

import jax.numpy as jnp
import jax.random as jr
from jax import tree

from .annotations import JaxArray

T = TypeVar('T')


class Projectable:
    """Base class to override how an element of a PyTree is projected."""
    def project(self, projector: Projector) -> Self:
        raise NotImplementedError


class Projector:
    """A tool to project the arrays in a PyTree to fewer dimensions. This is useful for graphing."""
    @override
    def __init__(self, *, seed: int = 0, dimensions: int = 2) -> None:
        super().__init__()
        self.seed = seed
        self.dimensions = dimensions
        self._projection_matrices: dict[int, JaxArray] = {}

    def project(self, projectable: T) -> T:
        """Project the arrays in a PyTree to the plane."""
        return tree.map(self._project, projectable, is_leaf=lambda x: isinstance(x, Projectable))

    def get_projection_matrix(self, features: int) -> JaxArray:
        """Produce a projection matrix."""
        if features <= self.dimensions:
            return jnp.eye(features)
        if features not in self._projection_matrices:
            self._projection_matrices[features] = self._random_directions(features)
        return self._projection_matrices[features]

    def _project(self, projectable: T) -> T:
        """Project an array or Projectable instance to the plane."""
        if projectable is None:
            return projectable
        if isinstance(projectable, Projectable):
            return projectable.project(self)

        assert isinstance(projectable, JaxArray)
        features = projectable.shape[-1]
        if features <= self.dimensions:
            return projectable
        projection_matrix = self.get_projection_matrix(features)
        return projectable @ projection_matrix  # type: ignore[return-value] # pyright: ignore

    def _random_directions(self, features: int) -> JaxArray:
        """Produce a random projection matrix.

        Returns: A matrix of random numbers with `features` rows and `self.dimensions` columns, each
            column of which is mutually orthogonal and unit-length.
        """
        if features < self.dimensions:
            raise ValueError
        if self.seed == 0 and features == self.dimensions:
            return jnp.eye(features)
        key = jr.key(self.seed)
        a = jr.uniform(key, shape=(features, self.dimensions))
        result = jnp.linalg.qr(a)
        assert not isinstance(result, JaxArray)
        return result.Q
