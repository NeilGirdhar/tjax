from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.lax import scan
from jax.ops import segment_sum

from .annotations import JaxArray, KeyArray
from .dataclasses.dataclass import dataclass

__all__ = ['cluster_with_k_means']


_k_means_dtype = jnp.int32


@partial(jit, static_argnames=('n_clusters',))
def _kmeans_plus_plus_init(key: KeyArray, points: JaxArray, n_clusters: int) -> JaxArray:
    # Initialize the centroid array.
    n_samples, n_features = points.shape
    centroids = jnp.empty((n_clusters, n_features))
    keys = jr.split(key, n_clusters)

    # Choose the first centroid randomly.
    first_idx = jr.randint(keys[0], shape=(), minval=0, maxval=n_samples)
    centroids = centroids.at[0].set(points[first_idx])

    for i, key_i in enumerate(keys[1:]):
        # Compute all distances between all centroids and all points.
        # Has shape (n_samples, n_clusters, n_features).
        distances = jnp.square(points[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :])
        distances = jnp.sum(distances, axis=2)  # Has shape (n_samples, n_clusters).

        # Compute distance to closest centroid from all points.
        distances = jnp.min(distances, axis=1)  # Has shape (n_samples,).

        # Sample a new centroid with probability proportional to the distances.
        p = distances / jnp.sum(distances)
        next_idx = jr.choice(key_i, a=n_samples, p=p)
        centroids = centroids.at[i].set(points[next_idx])
    return centroids


@dataclass
class _KMeansCarry:
    centroids: JaxArray
    cluster_assignments: JaxArray


def _k_means_single_iteration(points: JaxArray,
                              n_clusters: int,
                              carry: _KMeansCarry,
                              _: None
                              ) -> tuple[_KMeansCarry, None]:
    # Has shape (n_samples, n_clusters, n_features).
    distances = jnp.square(points[:, jnp.newaxis] - carry.centroids[jnp.newaxis])
    distances = jnp.sum(distances, axis=2)  # Has shape (n_samples, n_clusters).
    # Has shape (n_samples,)
    best_centroids = jnp.argmin(distances, axis=1).astype(_k_means_dtype)

    # total = jnp.zeros(n_clusters).at[best_centroids].add(points)
    # count = jnp.zeros(n_clusters).at[best_centroids].add(1.0)
    total = segment_sum(points, best_centroids, n_clusters)  # (n_clusters, n_features)
    count = segment_sum(1.0, best_centroids, n_clusters)  # (n_clusters,)
    centroids = total / count[slice(None), jnp.newaxis]
    return _KMeansCarry(centroids, best_centroids), None


@partial(jit, static_argnames=('iterations',))
def _k_means_many_iterations(points: JaxArray,
                             initial_means: JaxArray,
                             iterations: int
                             ) -> tuple[JaxArray, JaxArray]:
    n_samples = points.shape[0]
    n_clusters = initial_means.shape[0]
    cluster_assignments = jnp.zeros(n_samples, dtype=_k_means_dtype)
    carry = _KMeansCarry(initial_means, cluster_assignments)
    k_means_update = partial(_k_means_single_iteration, points, n_clusters)
    carry, _ = scan(k_means_update, carry, None, iterations)
    return carry.centroids, carry.cluster_assignments


def cluster_with_k_means(key: KeyArray,
                         points: JaxArray,
                         n_clusters: int,
                         iterations: int
                         ) -> JaxArray:
    """Cluster with k-means.

    Args:
        key: A random key.
        points: An array with shape (n_samples, n_features).
        n_clusters: The number of clusters to produce.
        iterations: The number of iterations to do.
    """
    initial_means = _kmeans_plus_plus_init(key, points, n_clusters)
    _, cluster_assignments = _k_means_many_iterations(points, initial_means, iterations)
    return cluster_assignments
