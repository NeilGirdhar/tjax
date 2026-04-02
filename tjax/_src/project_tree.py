from __future__ import annotations

from typing import Self, override

import jax.numpy as jnp
import jax.random as jr
from jax import tree

from .annotations import JaxArray


class Projectable:
    """Base class for objects that know how to project themselves via a :class:`Projector`.

    Subclasses should override :meth:`project` to implement custom dimensionality reduction
    instead of the default random-projection applied to leaf arrays.
    """

    def project(self, projector: Projector) -> Self:
        """Return a projected copy of this object using ``projector``."""
        raise NotImplementedError


class Projector:
    """Project the arrays in a pytree to a lower-dimensional space, useful for visualisation.

    Arrays with more features than ``dimensions`` are mapped to ``dimensions``
    dimensions via a random orthonormal projection matrix.  The same matrix is
    reused for every array with the same number of features, so relative
    distances are preserved across calls.
    """

    @override
    def __init__(self, *, seed: int = 0, dimensions: int = 2) -> None:
        super().__init__()
        self.seed = seed
        self.dimensions = dimensions
        self._projection_matrices: dict[int, JaxArray] = {}

    def project[T](self, projectable: T) -> T:
        """Project the arrays in a PyTree to the plane."""
        return tree.map(self._project, projectable, is_leaf=lambda x: isinstance(x, Projectable))

    def get_projection_matrix(self, features: int) -> JaxArray:
        """Return a ``(features, self.dimensions)`` projection matrix for arrays of that width.

        Results are cached by ``features`` so the same matrix is reused across
        multiple calls, ensuring consistency within a single :class:`Projector`
        instance.
        """
        if features <= self.dimensions:
            return jnp.eye(features)
        if features not in self._projection_matrices:
            self._projection_matrices[features] = self._random_directions(features)
        return self._projection_matrices[features]

    def _project[T](self, projectable: T) -> T:
        """Project an array or Projectable instance to the plane."""
        if projectable is None:
            return projectable
        if isinstance(projectable, Projectable):
            return projectable.project(self)  # type: ignore

        assert isinstance(projectable, JaxArray)
        features = projectable.shape[-1]
        if features <= self.dimensions:
            return projectable
        projection_matrix = self.get_projection_matrix(features)
        return projectable @ projection_matrix  # type: ignore # pyright: ignore

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
