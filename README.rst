=============
Tools for JAX
=============

.. role:: bash(code)
    :language: bash

.. role:: python(code)
   :language: python

This repository implements a variety of tools for the differential programming library
`JAX <https://github.com/google/jax>`_.  It includes:

- A dataclass decorator that facilitates defining JAX trees, provides convenient text display, and
  provides a mypy plugin

- A custom VJP decorator that supports both static and non-differentiable arguments

- A random number generator class

- JAX tree registration for `NetworkX <https://networkx.github.io/>`_ graph types

- Testing tools that automatically produce testing code

See the `documentation <https://neilgirdhar.github.io/tjax/tjax/index.html>`_.

Contribution guidelines
=======================

- Conventions: PEP8.

- How to run tests: :bash:`pytest .`

- How to clean the source:

  - :bash:`isort tjax`
  - :bash:`pylint tjax`
  - :bash:`mypy tjax`
  - :bash:`flake8 tjax`
