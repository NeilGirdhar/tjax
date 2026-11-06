[![PyPI - Version](https://img.shields.io/pypi/v/tjax)](https://pypi.org/project/tjax/)
[![EffVer Versioning](https://img.shields.io/badge/version_scheme-EffVer-0097a7)](https://jacobtomlinson.dev/effver)
[![SPEC-0](https://img.shields.io/badge/SPEC-0-blue)](https://scientific-python.org/specs/spec-0000/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tjax)

# Tools for JAX

This repository implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).

## Dataclasses

The [`dataclass`](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/dataclasses) decorator
defines a frozen dataclass that is also a JAX pytree, so you can pass your own structured objects to
JAX transformations. Fields are dynamic by default: JAX traces them, and new values reuse compiled
code. `field(static=True)` marks a field as static instead, which makes it part of the compiled
function, so Python can branch on it, and changing it recompiles.

```python
import copy

import jax
import jax.numpy as jnp
from jax import Array

from tjax.dataclasses import dataclass, field


@dataclass
class Line:
    slope: Array
    intercept: Array
    clip: bool = field(default=False, static=True)

    def __call__(self, x: Array) -> Array:
        y = self.slope * x + self.intercept
        return jnp.clip(y, 0.0, 1.0) if self.clip else y


@jax.jit
def loss(line: Line, x: Array, y: Array) -> Array:
    return jnp.sum(jnp.square(line(x) - y))


x = jnp.array([0.0, 1.0])
y = jnp.array([1.0, 3.0])
line = Line(jnp.asarray(1.0), jnp.asarray(0.0))

loss(line, x, y)  # Compiles.
loss(copy.replace(line, slope=jnp.asarray(2.0)), x, y)  # Reuses the compiled function.
loss(copy.replace(line, clip=True), x, y)  # Recompiles.

# Differentiating gives a Line of derivatives: slope=-4 and intercept=-6.
jax.grad(loss)(line, x, y)
```

### Alternatives

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

## Gradient transformations

The [gradient package](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/gradient) provides
every transformation in [optax](https://github.com/google-deepmind/optax) as a dataclass whose
fields are the transformation's arguments. Each transformation is a pytree, so JAX traces its
hyperparameters like any other array.

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

See the [gradient documentation](https://github.com/NeilGirdhar/tjax/blob/main/docs/gradient.md)
for its coverage of optax, the multi-loss transformations it adds, and its base classes.

## Pretty printing

[`print_generic`](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/display) displays
aggregate and vector types, including dataclasses, as a colorized tree, and it works on traced
values inside jitted functions. It formats small arrays as tables, and summarizes arrays that don't
fit:

```python
from tjax import print_generic

print_generic({"weights": jnp.arange(6.0).reshape(2, 3), "large": jnp.ones((200, 200))})
```

```text
dict
├── large=Jax Array (200, 200) float32
│   ├── mean=1.0
│   └── deviation=0.0
└── weights=Jax Array (2, 3) float32
    └──  0.0000 │ 1.0000 │ 2.0000
         3.0000 │ 4.0000 │ 5.0000
```

`display_generic` returns the same tree for you to format yourself.

## Other tools

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

## Contributing

Contributions are welcome. See the
[contribution guidelines](https://github.com/NeilGirdhar/tjax/blob/main/CONTRIBUTING.md).
