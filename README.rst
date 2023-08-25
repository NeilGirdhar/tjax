=============
Tools for JAX
=============

|pypi| |py_versions|

.. |pypi| image:: https://img.shields.io/pypi/v/tjax
   :   alt: PyPI - Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/tjax
   :   alt: PyPI - Python Version

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

This repository implements a variety of tools for the differential programming library
`JAX <https://github.com/google/jax>`_.

----------------
Major components
----------------

Tjax's major components are:

- A `dataclass <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/dataclasses>`_ decorator :python:`dataclass` that facilitates defining structured JAX objects (so-called "pytrees"), which benefits from:

  - the ability to mark fields as static (not available in `chex.dataclass`), and
  - a display method that produces formatted text according to the tree structure.

- A `fixed_point <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/fixed_point>`_ finding library heavily based on `fax <https://github.com/gehring/fax>`_.  Our
  library

  - supports stochastic iterated functions, and
  - uses dataclasses instead of closures to avoid leaking JAX tracers.

- A `shim <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/gradient>`_ for the gradient transformation library `optax <https://github.com/deepmind/optax>`_ that supports:


  - easy differentiation and vectorization of “gradient transformation” (learning rule) parameters,
  - gradient transformation objects that can be passed *dynamically* to jitted functions, and
  - generic type annotations.

- A pretty printer :python:`print_generic` for aggregate and vector types, including dataclasses.  (See
  `display <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/display>`_.)  It features:

  - a version for printing traced values :python:`tapped_print_generic`,
  - decoding size of the batched axes when printing ordinary and traced values,
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

- Basic tools like :python:`divide_where`.  (See `tools <https://github.com/NeilGirdhar/tjax/blob/master/tjax/_src/tools.py>`_.)

-----------------------
Contribution guidelines
-----------------------

- Conventions: PEP8.

- How to run tests: :bash:`pytest .`

- How to clean the source:

  - :bash:`ruff .`
  - :bash:`pyright`
  - :bash:`mypy`
  - :bash:`isort .`
  - :bash:`pylint tjax tests`
