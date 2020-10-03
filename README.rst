=============
Tools for JAX
=============

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

- A dataclass decorator :python:`dataclasss` that facilitates defining JAX trees, and has a MyPy plugin.
  (See `dataclass <https://github.com/NeilGirdhar/tjax/blob/master/tjax/dataclass.py>`_ and `mypy_plugin <https://github.com/NeilGirdhar/tjax/blob/master/tjax/mypy_plugin.py>`_.)

- A fixed point finding library heavily based on `fax <https://github.com/gehring/fax>`_.  Our
  library supports stochastic iterated functions, and avoids leaking JAX tracers.  (See
  `fixed_point <https://github.com/NeilGirdhar/tjax/blob/master/tjax/fixed_point>`_.)

----------------
Minor components
----------------

Tjax also includes:

- An object-oriented wrapper on top of `optax <https://github.com/deepmind/optax>`_.  (See
  `gradient <https://github.com/NeilGirdhar/tjax/blob/master/tjax/gradient>`_.)

- A pretty printer :python:`print_generic` for aggregate and vector types, including dataclasses.  (See
  `display <https://github.com/NeilGirdhar/tjax/blob/master/tjax/display.py>`_.)

- Versions of :python:`custom_vjp` and :python:`custom_jvp` that support both static and non-differentiable
  arguments.  (See `shims <https://github.com/NeilGirdhar/tjax/blob/master/tjax/shims.py>`_.)

- Tools for working with cotangents: :python:`copy_cotangent` and :python:`print_cotangent`.  (See
  `cotangent_tools <https://github.com/NeilGirdhar/tjax/blob/master/tjax/cotangent_tools.py>`_.)

- A random number generator class :python:`Generator`.  (See `generator <https://github.com/NeilGirdhar/tjax/blob/master/tjax/generator.py>`_.)

- JAX tree registration for `NetworkX <https://networkx.github.io/>`_ graph types.  (See
  `graph <https://github.com/NeilGirdhar/tjax/blob/master/tjax/graph.py>`_.)

- Leaky integration :python:`leaky_integrate` and Ornstein-Uhlenbeck process iteration
  :python:`diffused_leaky_integrate`.  (See `leaky_integral <https://github.com/NeilGirdhar/tjax/blob/master/tjax/leaky_integral.py>`_.)

- An improved version of :python:`jax.tree_util.Partial`.  (See `partial <https://github.com/NeilGirdhar/tjax/blob/master/tjax/partial.py>`_.)

- A Matplotlib trajectory plotter :python:`PlottableTrajectory`.  (See `plottable_trajectory <https://github.com/NeilGirdhar/tjax/blob/master/tjax/plottable_trajectory.py>`_.)

- A testing function :python:`assert_jax_allclose` that automatically produces testing code, and the related
  function :python:`jax_allclose`.  (See `testing <https://github.com/NeilGirdhar/tjax/blob/master/tjax/testing.py>`_.)

- Basic tools :python:`sum_tensors` and :python:`is_scalar`.  (See `tools <https://github.com/NeilGirdhar/tjax/blob/master/tjax/tools.py>`_.)

Also, see the `documentation <https://neilgirdhar.github.io/tjax/tjax/index.html>`_.

-----------------------
Contribution guidelines
-----------------------

- Conventions: PEP8.

- How to run tests: :bash:`pytest .`

- How to clean the source:

  - :bash:`isort tjax`
  - :bash:`pylint tjax`
  - :bash:`mypy tjax`
  - :bash:`flake8 tjax`
