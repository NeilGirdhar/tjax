.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

.. image:: https://img.shields.io/pypi/v/tjax
   :target: https://pypi.org/project/tjax/
   :alt: PyPI - Version
   :align: center
.. image:: https://img.shields.io/badge/version_scheme-EffVer-0097a7
   :alt: EffVer Versioning
   :target: https://jacobtomlinson.dev/effver
.. image:: https://img.shields.io/badge/SPEC-0-blue
   :target: https://scientific-python.org/specs/spec-0000/
   :alt: SPEC-0
   :align: center
.. image:: https://img.shields.io/pypi/pyversions/tjax
   :alt: PyPI - Python Version
   :align: center

=============
Tools for JAX
=============

This repository implements a variety of tools for the differential programming library
`JAX <https://github.com/google/jax>`_.

----------------
Major components
----------------

Tjax's major components are:

- A `dataclass <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/dataclasses>`_ decorator
  :python:`dataclass` that facilitates defining structured JAX objects (so-called "pytrees"), which
  benefits from:

  - the ability to mark fields as static (not available in `chex.dataclass`), and
  - a display method that produces formatted text according to the tree structure.

- A `shim <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/gradient>`_ for the gradient
  transformation library `optax <https://github.com/deepmind/optax>`_ that supports:


  - easy differentiation and vectorization of “gradient transformation” (learning rule) parameters,
  - gradient transformation objects that can be passed *dynamically* to jitted functions, and
  - generic type annotations.

- A pretty printer :python:`print_generic` for aggregate and vector types, including dataclasses.  (See
  `display <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/display>`_.)  It features:

  - support for traced values,
  - colorized tree output for aggregate structures, and
  - formatted tabular output for arrays (or statistics when there's no room for tabular output).

----------------
Minor components
----------------

Tjax also includes:

- Versions of :python:`custom_vjp` and :python:`custom_jvp` that support being used on methods:
  :python:`custom_vjp_method` and :python:`custom_vjp_method`
  (See `shims <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/shims.py>`_.)

- Tools for working with cotangents.  (See
  `cotangent_tools <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/cotangent_tools.py>`_.)

- JAX tree registration for `NetworkX <https://networkx.github.io/>`_ graph types.  (See
  `graph <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/graph.py>`_.)

- Leaky integration :python:`leaky_integrate` and Ornstein-Uhlenbeck process iteration
  :python:`diffused_leaky_integrate`.  (See `leaky_integral <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/leaky_integral.py>`_.)

- An improved version of :python:`jax.tree_util.Partial`.  (See `partial <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/partial.py>`_.)

- A testing function :python:`assert_tree_allclose` that automatically produces testing code.  And, a related
  function :python:`tree_allclose`.  (See `testing <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/testing.py>`_.)

- Basic tools like :python:`divide_where`.  (See `tools <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/math_tools.py>`_.)

-----------------------
Contribution guidelines
-----------------------

The implementation should be consistent with the surrounding style, be type annotated, and pass the
linters below.

To run tests: :bash:`pytest`

There are a few tools to clean and check the source:

- :bash:`ruff check`
- :bash:`pyright`
- :bash:`mypy`
- :bash:`isort .`
- :bash:`pylint tjax tests`
