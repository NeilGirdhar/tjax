"""
This library implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).
"""
from . import dataclasses, fixed_point, gradient
from ._src.abstract_method_decorators import JaxAbstractClass, abstract_custom_jvp, abstract_jit
from ._src.annotations import (Array, BooleanArray, BooleanNumeric, Complex, ComplexArray,
                               ComplexNumeric, Integral, IntegralArray, IntegralNumeric, JaxArray,
                               JaxBooleanArray, JaxComplexArray, JaxIntegralArray, JaxRealArray,
                               NumpyArray, NumpyBooleanArray, NumpyBooleanNumeric,
                               NumpyComplexArray, NumpyComplexNumeric, NumpyIntegralArray,
                               NumpyIntegralNumeric, NumpyRealArray, NumpyRealNumeric, PyTree, Real,
                               RealArray, RealNumeric, Shape, ShapeLike, SliceLike,
                               TapFunctionTransforms)
from ._src.cotangent_tools import (copy_cotangent, cotangent_combinator, print_cotangent,
                                   replace_cotangent, scale_cotangent)
from ._src.display import display_generic, print_generic, tapped_print_generic
from ._src.dtypes import (complex_dtype, default_atol, default_rtol, default_tols, float_dtype,
                          int_dtype)
from ._src.generator import vmap_split
from ._src.leaky_integral import (diffused_leaky_integrate, leaky_covariance, leaky_data_weight,
                                  leaky_integrate, leaky_integrate_time_series)
from ._src.partial import Partial
from ._src.plottable_trajectory import PlottableTrajectory
from ._src.shims import custom_jvp, custom_jvp_method, custom_vjp, custom_vjp_method, jit
from ._src.testing import (assert_tree_allclose, get_relative_test_string, get_test_string,
                           tree_allclose)
from ._src.tools import (abs_square, divide_nonnegative, divide_where, inverse_softplus, is_scalar,
                         zero_tangent_like)

__all__ = ['Array', 'BooleanArray', 'BooleanNumeric', 'Complex', 'ComplexArray', 'ComplexNumeric',
           'Integral', 'IntegralArray', 'IntegralNumeric', 'NumpyArray', 'NumpyBooleanArray',
           'NumpyIntegralArray', 'NumpyRealArray', 'NumpyComplexArray', 'NumpyBooleanNumeric',
           'NumpyIntegralNumeric', 'NumpyRealNumeric', 'NumpyComplexNumeric', 'Partial',
           'PlottableTrajectory', 'PyTree', 'Real', 'RealArray', 'RealNumeric', 'Shape',
           'ShapeLike', 'SliceLike', 'TapFunctionTransforms', 'abs_square', 'assert_tree_allclose',
           'copy_cotangent', 'scale_cotangent', 'custom_jvp', 'custom_jvp_method', 'custom_vjp',
           'custom_vjp_method', 'dataclasses', 'default_atol', 'default_rtol', 'default_tols',
           'int_dtype', 'float_dtype', 'complex_dtype', 'diffused_leaky_integrate',
           'display_generic', 'divide_nonnegative', 'divide_where', 'fixed_point',
           'get_relative_test_string', 'get_test_string', 'gradient', 'tapped_print_generic',
           'inverse_softplus', 'is_scalar', 'jit', 'leaky_covariance', 'leaky_data_weight',
           'leaky_integrate', 'leaky_integrate_time_series', 'cotangent_combinator',
           'print_cotangent', 'print_generic', 'replace_cotangent', 'tree_allclose', 'vmap_split',
           'zero_tangent_like', 'JaxArray', 'JaxBooleanArray', 'JaxIntegralArray', 'JaxRealArray',
           'JaxComplexArray', 'JaxAbstractClass', 'abstract_jit', 'abstract_custom_jvp']
#
# __pdoc__ = {}
# __pdoc__['PyTreeLike'] = False
# __pdoc__['Field'] = False
# __pdoc__['InitVar'] = False
# __pdoc__['FrozenInstanceError'] = False
#
# document_dataclass(__pdoc__, 'Partial')
# del document_dataclass
