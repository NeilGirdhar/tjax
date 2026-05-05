# Changelog

This changelog summarizes TJAX releases inferred from version changes in `pyproject.toml`.
Each section covers changes since the previous release.

## 1.5.0 - 2026-05-04

- Added `negate_cotangent` and `scale_tree`.
- Removed `cotangent_combinator` and `reverse_scale_cotangent`.
- Moved `zero_from_primal` to `tree_tools`.
- Updated to JAX 0.10 and migrated type-ignore comments from pyright/`type: ignore` to ty/`ty: ignore`.

## 1.4.10 - 2026-04-15

- Added tests for the `HvpGradientTransformation`/`DiagHessianGradientTransformation`.
- Renamed shim.
- Tested `frozendict` display.
- Fixed the README and added a PyPI badge.
- Updated the GitHub release workflow.

## 1.4.0 - 2025-09-10

- Supported Python 3.14 and dropped Python 3.11.
- Added `bessel_iv_ratio` and `log_bessel_ive` (pure-JAX Bessel function implementations).
- Added `frozendict` with JAX JIT registration.
- Renamed `HessianGradientTransformation` to `HvpGradientTransformation` and added
  `DiagHessianGradientTransformation`.
- Removed `tjax.Partial` in favour of the standard-library equivalent.
- Switched type checking from Pyright to ty.
- Updated to JAX 0.8 and JAX 0.9.
- Switched to the uv build backend and added a PyPI release workflow.
- Enabled additional Ruff rules and sorted imports.

## 1.3.0 - 2025-05-21

- Added `Timer` and `display_time` for wall-clock profiling.
- Added `tree_sum` and `element_count`.
- Added `stop_gradient` (math-tools helper wrapping `jax.lax.stop_gradient`).
- Added `RngStream.fork` and made `RngStream.count` private.
- Added `Projector` and `Projectable` pytree-projection helpers.
- Added `GenericString` for display of generic string values.
- Removed k-means utilities.
- Corrected `stop_gradient` behaviour.
- Updated for JAX 0.6.1.
- Switched to Lefthook and dependency groups.
- Enabled ANN401 Ruff rule; adopted `xp.astype` from the Array API.
- Added `as_leaves` option to `print_generic`.
- Dropped Flax support.

## 1.2.0 - 2025-02-28

- Added `log_softplus` and `sublinear_softplus`.
- Added `stop_gradient` to the math-tools API.
- Switched math tools to use Array API types.
- Made `cast_to_result_type` use the Array API namespace.
- Removed `divide_nonnegative`, `leaky_integrate_time_series`, and `leaky_covariance`.
- Replaced `Any` with `object` in annotations throughout.
- Simplified `create_diagonal_array` to use JAX.
- Updated for SPEC 0 and switched to `array_namespace`.
- Added dev optional dependency group and pre-commit configuration.
- Adopted `jax.random` import alias `jr`.
- Fixed Flax 0.10.3 compatibility errors.
- Added meta-learning and recompilation examples.

## 1.1.0 - 2024-11-24

- Updated for Flax NNX 0.10.
- Added `zero_from_primal` to cotangent tools.
- Added `normalize` and L1-normalization to math tools.
- Removed `zero_tangent_like`.
- Verified compatibility with JAX 0.4.35.
- Polished Flax object display in `display_generic`.
- Added `pre-commit` configuration.
- Updated license metadata and lock files.

## 1.0.0 - 2024-08-19

- Added `RngStream`, `create_streams`, `fork_streams`, and `sample_streams` for structured
  RNG management.
- Added `result_type` and `cast_to_result_type` for Array API dtype promotion.
- Added `dynamic_tree_all`; removed `tree_map_with_path`.
- Expanded Optax optimizer aliases to cover the full Optax transform set.
- Split the graph module into submodules (`display`, `register_jax`, `register_flax`, `types_`).
- Made graph flatteners use proper JAX key types.
- Switched dataclass field metadata from `pytree_node` to `static` (Equinox-compatible naming).
- Removed `tjax.fixed_point` in favour of Optimistix.
- Added matrix-tools helpers (`matrix_dot_product`, `matrix_vector_mul`, `outer_product`).
- Fixed Flax object display in `display_generic`.
- Switched packaging to uv/hatch.
- Updated license to current metadata.

## 0.32.0 - 2024-05-07

- Used `jax.register_dataclass` for pytree registration, replacing the manual implementation.
- Added a Hessian shim (`tjax.hessian`).
- Generalized `create_diagonal_array` to work with JAX arrays.
- Upgraded to NumPy 2.
- Added Array API usage in cotangent tools and the fixed-point module.
- Supported Flax 0.8.4 and fixed Flax import guards.
- Updated to JAX 0.4.26 and JAX 0.4.27 in the patch line.
- Supported `init=False` in dataclass fields.
- Fixed Optax 1.8 deprecation warning; showed seen arrays in display.
- Added `show_seen_arrays` to `print_generic`.
- Blocked seen-set ellision for type objects.
- Fixed array API misuse and fixed-point stochastic annotation.
- Updated to Ruff 0.2 and switched to local PyRight environment.

## 0.31.0 - 2024-01-21

- Updated to Flax 0.8 and improved display of NNX components.
- Adopted Array API usage in cotangent tools and the fixed-point module.
- Updated to JAX 0.4.26 and JAX 0.4.27.
- Re-organized imports and sorted `__all__`.
- Fixed mutable-default and Pyright errors.
- Switched to Ruff 0.2.
- Added a default value for the `seen` parameter in `print_generic`.
- Updated Flax to 0.8.3; prepared the NNX integration.

## 0.30.0 - 2023-10-09

- Supported Python 3.12 and added `KeyArray` type alias.
- Added `ZeroIteratedFunction`.
- Removed `PlottableTrajectory`.
- Exposed `DataclassInstance`.
- Removed `vmap_split`.
- Fixed batch-dimension display for ordinary classes.
- Improved `PyTreeDef` output in `print_generic`.
- Added `complex_dtype`, `float_dtype`, and `int_dtype` helpers.
- Made `jit` preserve `__override__`.
- Supported applying JAX decorators to abstract methods.
- Removed dtype caches.
- Added `augment_cotangent` to cotangent tools (late in the cycle).
- Added `numpy_tools` module with `create_diagonal_array`.
- Improved graph flattening and fixed undirected graph serialization.
- Blocked dunder method display and made some internals private.
- Updated Ruff, Pyright, and Poetry lock.

## 0.29.0 - 2023-04-18

- Dropped Python 3.8.
- Adopted `typing_extensions.override`.
- Removed pycodestyle in favour of Ruff.
- Made Pyright report unnecessary type-ignore comments.
- Updated KeyArray display tests.
- Updated Poetry lock and polished packaging.

## 0.28.0 - 2023-02-06

- Added JAX array types (`JaxRealArray`, `JaxComplexArray`, etc.) and tightened annotations.
- Used `register_pytree_with_keys` for richer JAX tree metadata.
- Updated to JAX 0.4.6.
- Updated Poetry lock for newer JAX.

## 0.27.0 - 2023-02-28

- Split `custom_jvp` and `custom_vjp` shims into special-method variants (`custom_jvp_method`,
  `custom_vjp_method`).
- Switched tests and internal usage to `tjax.jit`.
- Removed the MyPy plugin; upgraded to MyPy 1.0.
- Updated Poetry lock and polished annotations.

## 0.26.0 - 2023-02-06

- Switched to Ruff for linting.
- Updated to MyPy 1.0 and adopted `Self`.
- Prepared the Array API constrained interface.
- Improved `tapped_print_generic` handling of pytrees.
- Exposed `GenericGradientState` and `ChainedGradientState`.
- Corrected `field_specifiers` naming.
- Fixed `get_test_string` for integers and namedtuples.
- Added support for Python 3.11 lock.

## 0.25.0 - 2022-12-18

- Updated for JAX 0.4.
- Switched color library from `colorful` to `rich` for `display_generic`.
- Reorganized the display package and added a string `display_generic` dispatch.
- Added `solarized` color scheme support.
- Added `BatchDimensions` type alias and improved batched-dimension display.
- Fixed bugs with `tapped_print_generic` for dicts and batch tracers.
- Supported Python 3.11.
- Replaced `np.prod` with `math.prod`.
- Constrained the interface to prepare for the Array API.

## 0.24.0 - 2022-12-13

- Updated for JAX 0.4.
- Supported Python 3.11.
- Updated README for `print_generic`.

## 0.23.0 - 2022-11-30

- Added `scale_cotangent`, `reverse_scale_cotangent`, and `aux_cotangent_scales` to
  `cotangent_combinator`.
- Added `cotangent_combinator`.
- Added `frozen_default` to `dataclass_transform`.
- Added assertions to `divide_where` and cotangent tools.
- Removed the deprecated `Generator`.
- Fixed JAX annotation compatibility and removed use of `jax._src`.
- Added `result` return to `id_display`.
- Allowed `inverse_softplus` to accept `RealNumeric`.

## 0.22.0 - 2022-09-28

- Adopted `jax.Array` throughout and removed `tjax.Generator`.
- Exposed `vmap_split` directly at the top level.
- Stopped using `tjax.Generator` internally.

## 0.21.0 - 2022-09-14

- Displayed mean and variance of large arrays in `display_generic`.
- Deprecated `Generator` (pending removal); exposed `vmap_split` at the top level.
- Updated Poetry lock.

## 0.20.0 - 2022-08-22

- Added `inverse_softplus`.
- Removed the Flax dependency.
- Updated Poetry lock.

## 0.19.0 - 2022-02-13

- Updated to JAX 0.3.14.
- Removed yapf code formatting in favour of other tooling.
- Removed `tjax.dataclass_transform` from the top-level namespace.
- Reorganized dataclass internals and moved helper modules.
- Added `as_shallow_dict` to the dataclass API.
- Used `ParamSpec` in `custom_jvp` and `custom_vjp` shims.
- Fixed `diffused_leaky_integrate` annotations.
- Repaired the `Partial` class.
- Registered "key paths" for `tjax.dataclass` instances (Jupyter / debugger UX).
- Removed unnecessary type-ignore directives.

## 0.18.0 - 2021-12-21

- Updated to JAX 0.3 and simplified dataclass pytree registration.
- Added Flax dependency and graph-to-Flax state registration.
- Added `dataclass_transform` for working with Flax dataclasses.
- Supported displaying all dataclasses and standard Python modules in `display_generic`.
- Adopted `__dataclass_transform__` for static type-checker support.
- Added `zero_tangent` (renamed `zero_tangent_like` in the patch line).
- Updated to NumPy 1.22 and removed matplotlib dependency.
- Switched to Pyright for type checking.
- Dropped support for Python 3.7.

## 0.17.0 - 2021-12-14

- Added `SimpleScan` implementing `IteratedFunctionBase`.
- Added type annotation overloads.
- Used Pyright for type checking.

## 0.16.0 - 2021-11-24

- Supported Python 3.10.
- Simplified dataclass interface to `dataclass` and `field`; added `kw_only` support.
- Removed `CotangentMapper` and `block_cotangent` (superseded by `jax.lax.stop_gradient`).
- Switched to `pyproject-flake8`.

## 0.15.0 - 2021-11-20

- Added a `GradientState` base class.
- Added the remaining Optax alias transforms (full Optax coverage).

## 0.14.0 - 2021-10-13

- Added all Optax gradient transformations.
- Added `raise_on_nan` to `print_generic`.
- Added the `no_jvp` option to `id_display`.
- Added RMSprop gradient transformation.
- Delegated to the MyPy dataclasses plugin.
- Removed `dataclass.tree_unflatten`, `tree_flatten`, and `replace`; switched to the
  `dataclasses.replace` standard function.
- Fixed typing errors and repaired tests.

## 0.13.0 - 2021-07-14

- Renamed `assert_jax_allclose` to `assert_tree_allclose` and `jax_allclose` to `tree_allclose`.
- Replaced `nonstatic` with `dynamic` in dataclass field metadata.
- Upgraded `print_cotangent` to display batch dimensions.
- Updated to NumPy 1.21.
- Added overloads for `abs_square`.
- Enabled MyPy warnings.

## 0.12.0 - 2021-06-23

- Updated for NumPy 1.21.
- Added overloads and improved type annotations.
- Fixed test infrastructure and improved testing code assertions.

## 0.11.0 - 2021-05-12

- Changed `Generator` to no longer implicitly split on each call.
- Fixed initialization of the minimum-ATOL estimator.
- Enabled MyPy warnings.

## 0.10.0 - 2021-04-22

- Added `divide_where` and renamed `divide_nonnegative`.
- Added `block_variable_cotangent`.
- Added `CotangentMapper`.
- Added `label_function`, `simple_label`, and `clip_boolean` to `PlottableTrajectory`.
- Updated JAX `custom_vjp` type annotations.

## 0.9.0 - 2021-04-12

- Reorganized the package: moved implementation into `_src` and hid internal modules.
- Added `BoolArray` and `IntegerArray` type aliases; removed `vjp_with_aux`.
- Improved display of `DeviceArray` objects.
- Re-engineered type annotations throughout.
- Enabled `jax.experimental.enable_x64`.

## 0.8.0 - 2021-02-16

- Added higher-order fixed-point differentiation support.
- Added docstrings to the fixed-point module.

## 0.7.0 - 2020-11-15

- Added `Stochastic Meta-Descent` (SMD) gradient transformation.
- Added `SecondOrderGradientTransformation`.
- Added `PlottableTrajectory` with filtering and plotting utilities.
- Added `zero_out_cotangent` and cotangent tools (`replace_cotangent`, `transform_cotangent`).
- Added support for displaying mappings and dynamic JAX tracers.
- Updated `Generator` for JAX 0.1.64.
- Generalized cotangent transformation to `transform_cotangent`.
- Made `minimum_iterations` a required parameter.
- Added Optax `adamw` integration.
- Supported NumPy 1.20.

## 0.6.0 - 2020-10-16

- Added cotangent-manipulation utilities (`block_cotangent`, `copy_cotangent`, `print_cotangent`).
- Added `vjp_with_aux` type annotation stub.
- Swapped the output order of `Generator` sampling methods.
- Added `zero_out_cotangent`.
- Updated Generator for JAX 0.1.64.
- Added `setuptools` to the build system.

## 0.5.0 - 2020-09-30

- Supported JAX 0.2.
- Added a MyPy plugin for `tjax.dataclass`.
- Added trajectory-handling generalization.
- Moved `Generator` into the stochastic state module.
- Removed `cooperative_dataclasses` and the meta-parameter implementation.
- Added a patch for dataclass defaulted fields.

## 0.4.0 - 2020-08-18

- Added `PlottableTrajectory` and `print_cotangent`.
- Added Bernoulli and uniform distributions.
- Added gradient-transformation reorganization.
- Renamed `Parameters` to `Weights`; renamed `hashed_fields` to `static_fields`.
- Added metaparameter support and `field_names_and_values`.
- Provided `dataclasses.fields` access.
- Updated for the new JAX `id_tap` interface.

## 0.3.0 - 2020-07-29

- Added a fixed-point iteration library.
- Added `int_dtype` and support for Python 3.7.
- Added `jit` shim that propagates `__isabstractmethod__`.
- Improved testing string utilities and display.
- Added `Partial` class and `py.typed` marker.
- Supported displaying lists, tuples, and graphs.

## 0.2.0 - 2020-07-08

- Added `RealTensor` and `ComplexTensor` type aliases.
- Added Optax-based optimizers.
- Used `chex.Array`.
- Added `Generator.vmap_split`.
- Displayed `BatchTracer` objects.
- Allowed editable installs via dephell.

## 0.1.0 - 2020-07-08

- Initial release: transferred code from an earlier project.
- Provided `tjax.dataclass` (JAX-compatible dataclasses with static-field support).
- Provided type annotations (`RealArray`, `ComplexArray`, `PyTree`, `Shape`, etc.).
- Provided `Generator` for JAX random-number management.
- Provided `custom_jvp`, `custom_vjp`, and `jit` shims.
- Provided `assert_jax_allclose` and `get_test_string` testing utilities.
- Provided leaky-integral utilities and graph display.
