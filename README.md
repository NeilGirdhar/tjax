[![PyPI - Version](https://img.shields.io/pypi/v/tjax)](https://pypi.org/project/tjax/)
[![EffVer Versioning](https://img.shields.io/badge/version_scheme-EffVer-0097a7)](https://jacobtomlinson.dev/effver)
[![SPEC-0](https://img.shields.io/badge/SPEC-0-blue)](https://scientific-python.org/specs/spec-0000/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tjax)

# Tools for JAX

This repository implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).

## Major components

Tjax's major components are:

- A [dataclass](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/dataclasses) decorator
  `dataclass` that facilitates defining structured JAX objects (so-called "pytrees"), which
  benefits from:
    - the ability to mark fields as static (not available in `chex.dataclass`), and
    - a display method that produces formatted text according to the tree structure.
- A [gradient package](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/gradient) that
  provides every transformation in [optax](https://github.com/google-deepmind/optax) as a pytree.
  (See [Gradient transformations](#gradient-transformations).)
- A pretty printer `print_generic` for aggregate and vector types, including dataclasses. (See
  [display](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/display).) It features:
    - support for traced values,
    - colorized tree output for aggregate structures, and
    - formatted tabular output for arrays (or statistics when there's no room for tabular output).

## Gradient transformations

`tjax.gradient` provides every optax transformation as a dataclass whose fields are the
transformation's arguments. Each transformation is a pytree, so JAX traces its hyperparameters like
any other array.

This matters because an optax transformation is a pair of functions that close over their
hyperparameters, which JAX can't see. You can't pass an optax transformation to a jitted function;
you construct it inside the traced function or wrap it with `optax.inject_hyperparams`. A tjax
transformation instead behaves like data:

- **Jit.** A transformation can be an argument of a jitted function, and a new value of a numeric
  hyperparameter, such as the learning rate, reuses the compiled function.
- **Differentiation.** `jax.grad` differentiates through an update with respect to its
  hyperparameters, for example to tune a learning rate by gradient descent.
- **Vectorization.** `jax.vmap` maps an update over hyperparameter values, for example to sweep
  learning rates in one call.
- **Static options.** Structural arguments, such as `nesterov` and weight-decay masks, are static
  fields, so jit compiles a specialized function for each setting.
- **Types.** `GradientTransformation[State, Weights]` is generic in its state and weight types, so a
  type checker can check what you pass to `update`.
- **Display.** A transformation prints with its hyperparameters, for example
  `Adam(learning_rate=0.01, b1=0.9, ...)`.

For example, this code compiles one Adam step for two learning rates, then differentiates and maps
the loss after one SGD step over the learning rate:

```python
import jax
import jax.numpy as jnp
import optax

import tjax.gradient as tg

params = {"w": jnp.array([1.0, -2.0])}


def loss(params):
    return jnp.sum(params["w"] ** 2)


@jax.jit
def loss_after_step(transform, params):
    gradient = jax.grad(loss)(params)
    updates, _ = transform.update(gradient, transform.init(params), params)
    return loss(optax.apply_updates(params, updates))


loss_after_step(tg.Adam(1e-2), params)  # Compiles.
loss_after_step(tg.Adam(3e-2), params)  # Reuses the compiled function.

# Differentiate with respect to the learning rate.
jax.grad(lambda lr: loss_after_step(tg.SGD(lr), params))(0.1)

# Take one step at each of three learning rates.
jax.vmap(lambda lr: loss_after_step(tg.SGD(lr), params))(jnp.array([0.01, 0.1, 0.4]))
```

### Coverage of optax

`tjax.gradient` has everything optax offers for transforming gradients: every optimizer,
transformation, and combinator, including those in `optax.contrib`. Each wrapper takes the same
arguments as its optax counterpart, so `optax.adam(1e-2)` becomes `tg.Adam(1e-2)`. The test suite
checks every wrapper's parameters against optax's signatures, and its updates against optax's under
jit.

The remaining differences are deliberate:

- `optax.chain` is `ChainedGradientTransformation`, which takes a list of transformations.
- `optax.inject_hyperparams` has no wrapper because it isn't needed: hyperparameters are already
  traced, so you can replace or differentiate them directly.
- `optax.with_extra_args_support` has no wrapper because every `update` method accepts extra keyword
  arguments, such as `value`.
- `optax.optimistic_adam` is deprecated, so `OptimisticAdam` wraps its replacement,
  `optimistic_adam_v2`.

Everything else in optax works alongside tjax, such as `optax.apply_updates`.

### Transformations beyond optax

These have no optax equivalent:

- `SMDGradient` implements stochastic meta-descent, which adapts a learning rate for each weight
  using Hessian-vector products.
- `MTAdam` is Adam for several losses. It balances them automatically, so you don't need to tune
  loss weights. `MultiLossGradientTransformation` is the base class for such transformations,
  which receive each loss's gradient separately.
- `GradientCombiner` is the base class for multi-loss methods that decide only how to combine
  gradients, and it passes the result to an optimizer of your choice. Its subclasses are:
    - `GradientStatisticsAnnealing`, which weights each loss against a reference gradient.
    - `IMTLG`, which chooses weights so that the result favours no loss.
    - `PCGrad`, which removes the components of each gradient that conflict with the others.
    - `CAGrad`, which improves the worst-off loss while staying near the average gradient.
    - `GradVac`, which steers each pair of gradients toward the similarity they usually have.
- `HvpGradientTransformation` and `DiagHessianGradientTransformation` are base classes for update
  rules that receive a Hessian-vector product or the Hessian's diagonal.

### Interoperating with optax

- `to_optax` converts a tjax transformation into an optax one, and `FromOptax` does the reverse.
- `Schedule` wraps an optax schedule so that it can be passed as a learning rate.

## Dataclass alternatives

Several libraries offer dataclass-like pytrees. Compared with them, `tjax.dataclass`:

- **vs `chex.dataclass`:** chex can't mark fields static, which rules out a huge class of
  designs: flags, shapes, dtypes, mode strings, callable options, anything that should drive
  recompilation rather than live in the trace. Tjax marks them inline with `field(static=True)`.
- **vs `jax.tree_util.register_dataclass`:** the primitive forces a second call per class and
  wants the static and dynamic partition as parallel name lists at the registration site —
  awkward, duplicative, and easy to desync from the fields. Tjax does it in the decorator with
  `field(static=True)` on the field itself.
- **vs `flax.struct.dataclass`:** near-equivalent in capability; the win is not dragging Flax in
  for one decorator.
- **vs `equinox.Module`:** too heavyweight for just wanting a pytree dataclass. The `Module`
  base and metaclass constrain inheritance, and the idiomatic path pushes you into
  `eqx.filter_jit` / `eqx.partition` at every call site instead of declaring the static and
  dynamic boundary once on the class.

## Minor components

Tjax also includes:

- Versions of `custom_vjp` and `custom_jvp` that support being used on methods:
  `custom_vjp_method` and `custom_jvp_method`
  (See [shims](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/shims.py).)
- Tools for working with cotangents. (See
  [cotangent_tools](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/cotangent_tools.py).)
- JAX tree registration for [NetworkX](https://networkx.github.io/) graph types. (See
  [graph](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/graph/).)
- A hashable immutable mapping `frozendict` that is registered as a JAX pytree, so it can be
  passed through transforms like `jit` and `vmap`. (See
  [frozendict](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/frozendict.py).)
- Leaky integration `leaky_integrate` and Ornstein-Uhlenbeck process iteration
  `diffused_leaky_integrate`. (See
  [leaky_integral](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/leaky_integral.py).)
- A testing function `assert_tree_allclose` that automatically produces testing code. And, a related
  function `tree_allclose`, which compares pytrees with matching structure. (See
  [testing](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/testing.py).)
- Basic tools like `divide_or_zero` and `divide_where`. (See
  [tools](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/math_tools.py).)

## Contribution guidelines

The implementation should be consistent with the surrounding style, be type annotated, and pass the
linters below.

To run tests: `uv run pytest`

There are a few tools to clean and check the source:

- `uv run ruff check`
- `uv run ruff format`
- `uv run ty check`
