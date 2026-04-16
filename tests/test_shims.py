from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tjax._src.shims import hessian  # noqa: PLC2701


def _quad_with_aux(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """f(x) = sum(x**2), aux = x[0].  Hessian = 2*I."""
    return jnp.sum(x**2), x[0]


def test_hessian_has_aux_reverse_only() -> None:
    """Hessian with has_aux=True and reverse_only=True should return (H, aux)."""
    x = jnp.array([1.0, 2.0])
    h_rev, aux_rev = hessian(_quad_with_aux, has_aux=True, reverse_only=True)(x)
    assert_allclose(h_rev, 2.0 * np.eye(2), atol=1e-6)
    assert_allclose(aux_rev, x[0], atol=1e-6)


def test_hessian_has_aux_forward_over_reverse() -> None:
    """Hessian with has_aux=True (default forward-over-reverse) should return (H, aux)."""
    x = jnp.array([1.0, 2.0])
    h_fwd, aux_fwd = hessian(_quad_with_aux, has_aux=True, reverse_only=False)(x)
    assert_allclose(h_fwd, 2.0 * np.eye(2), atol=1e-6)
    assert_allclose(aux_fwd, x[0], atol=1e-6)


def test_hessian_has_aux_both_modes_agree() -> None:
    """Forward-over-reverse and double-reverse should give identical results with has_aux=True."""
    x = jnp.array([3.0, -1.0, 0.5])
    h_fwd, aux_fwd = hessian(_quad_with_aux, has_aux=True, reverse_only=False)(x)
    h_rev, aux_rev = hessian(_quad_with_aux, has_aux=True, reverse_only=True)(x)
    assert_allclose(h_rev, h_fwd, atol=1e-6)
    assert_allclose(aux_rev, aux_fwd, atol=1e-6)
