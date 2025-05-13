"""Avoiding recompilation.

Using tjax.gradient is much more efficient than optax since we can pass gradient transformation
objects without inducing unnecessary recompilation.
"""
from functools import partial
from typing import Any

from jax import jit
from optax import GradientTransformation as OptaxGradientTransformation
from optax import adam

from tjax.gradient import Adam, GradientTransformation


@partial(jit, static_argnames=('optax_gt',))
def try_model_optax(optax_gt: OptaxGradientTransformation) -> None:
    print("Recompiling using optax!")  # noqa: T201


@jit
def try_model_tjax(tjax_gt: GradientTransformation[Any, Any]) -> None:
    print("Recompiling using tjax!")  # noqa: T201


for learning_rate in [0.1, 0.2, 0.4, 0.8]:
    # Typically, the model and gradient transformations are created here.
    optax_gt = adam(learning_rate)
    tjax_gt = Adam(learning_rate)
    # Now, you would create the gradient state.

    # Then, these are passed to a jitted inference function.
    try_model_optax(optax_gt)
    try_model_tjax(tjax_gt)

    # With the result of infrence, you would update the grdient state.

# Prints:
# Recompiling using optax!
# Recompiling using tjax!
# Recompiling using optax!
# Recompiling using optax!
# Recompiling using optax!

# To avoid recompilation in Optax, you are forced to create the gradient transformation again inside
# the inference function.
