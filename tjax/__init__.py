"""
This library implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).
"""
from . import dataclasses, fixed_point, gradient
from ._src.dataclasses import dataclass
from ._src.annotations import (Array, BoolArray, ComplexArray, IntegerArray, PyTree, RealArray,
                               Shape, ShapeLike, SliceLike, TapFunctionTransforms)
from ._src.cotangent_tools import (block_cotangent, copy_cotangent, print_cotangent,
                                   replace_cotangent)
from ._src.display import display_generic, print_generic
from ._src.dtypes import (complex_dtype, default_atol, default_rtol, default_tols, int_dtype,
                          real_dtype)
from ._src.generator import Generator
from ._src.leaky_integral import (diffused_leaky_integrate, leaky_covariance, leaky_data_weight,
                                  leaky_integrate, leaky_integrate_time_series)
from ._src.partial import Partial
from ._src.plottable_trajectory import PlottableTrajectory
from ._src.shims import custom_jvp, custom_vjp, jit
from ._src.testing import (assert_jax_allclose, get_relative_test_string, get_test_string,
                           jax_allclose)
from ._src.tools import abs_square, is_scalar, safe_divide, sum_tensors

__all__ = ['Array', 'BoolArray', 'ComplexArray', 'Generator', 'IntegerArray', 'Partial',
           'PlottableTrajectory', 'PyTree', 'RealArray', 'Shape', 'ShapeLike', 'SliceLike',
           'TapFunctionTransforms', 'abs_square', 'assert_jax_allclose', 'block_cotangent',
           'complex_dtype', 'copy_cotangent', 'custom_jvp', 'custom_vjp', 'dataclass',
           'dataclasses', 'default_atol', 'default_rtol', 'default_tols',
           'diffused_leaky_integrate', 'display_generic', 'fixed_point', 'get_relative_test_string',
           'get_test_string', 'gradient', 'int_dtype', 'is_scalar', 'jax_allclose', 'jit',
           'leaky_covariance', 'leaky_data_weight', 'leaky_integrate',
           'leaky_integrate_time_series', 'print_cotangent', 'print_generic', 'real_dtype',
           'replace_cotangent', 'safe_divide', 'sum_tensors']
#
# __pdoc__ = {}
# __pdoc__['real_dtype'] = False
# __pdoc__['complex_dtype'] = False
# __pdoc__['PyTreeLike'] = False
# __pdoc__['Field'] = False
# __pdoc__['InitVar'] = False
# __pdoc__['FrozenInstanceError'] = False
#
# document_dataclass(__pdoc__, 'Generator')
# document_dataclass(__pdoc__, 'Partial')
# del document_dataclass
