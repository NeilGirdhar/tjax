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
- A [shim](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/gradient) for the gradient
  transformation library [optax](https://github.com/deepmind/optax) that supports:
    - easy differentiation and vectorization of “gradient transformation” (learning rule)
      parameters,
    - gradient transformation objects that can be passed *dynamically* to jitted functions, and
    - generic type annotations.
- A pretty printer `print_generic` for aggregate and vector types, including dataclasses. (See
  [display](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/display).) It features:
    - support for traced values,
    - colorized tree output for aggregate structures, and
    - formatted tabular output for arrays (or statistics when there's no room for tabular output).

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
- Basic tools like `divide_where`. (See
  [tools](https://github.com/NeilGirdhar/tjax/blob/main/tjax/_src/math_tools.py).)

## Contribution guidelines

The implementation should be consistent with the surrounding style, be type annotated, and pass the
linters below.

To run tests: `uv run pytest`

There are a few tools to clean and check the source:

- `uv run ruff check`
- `uv run ruff format`
- `uv run ty check`
