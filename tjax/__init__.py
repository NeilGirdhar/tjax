"""Tools for the differential programming library [JAX](https://github.com/google/jax)."""
from . import dataclasses, fixed_point, gradient
from ._src.abstract_method_decorators import JaxAbstractClass, abstract_custom_jvp, abstract_jit
from ._src.annotations import (Array, BooleanArray, BooleanNumeric, Complex, ComplexArray,
                               ComplexNumeric, Integral, IntegralArray, IntegralNumeric, JaxArray,
                               JaxBooleanArray, JaxComplexArray, JaxIntegralArray, JaxRealArray,
                               KeyArray, NumpyArray, NumpyBooleanArray, NumpyBooleanNumeric,
                               NumpyComplexArray, NumpyComplexNumeric, NumpyIntegralArray,
                               NumpyIntegralNumeric, NumpyRealArray, NumpyRealNumeric, PyTree, Real,
                               RealArray, RealNumeric, Shape, ShapeLike, SliceLike,
                               TapFunctionTransforms)
from ._src.cotangent_tools import (copy_cotangent, cotangent_combinator, print_cotangent,
                                   replace_cotangent, scale_cotangent)
from ._src.display.display_generic import display_generic
from ._src.display.internal import internal_print_generic
from ._src.display.print_generic import print_generic
from ._src.dtypes import default_atol, default_rtol, default_tols
from ._src.graph import (graph_arrow, graph_edge_name, register_graph_as_jax_pytree,
                         register_graph_as_nnx_node)
from ._src.leaky_integral import (diffused_leaky_integrate, leaky_covariance, leaky_data_weight,
                                  leaky_integrate, leaky_integrate_time_series)
from ._src.math_tools import (abs_square, divide_nonnegative, divide_where, inverse_softplus,
                              matrix_dot_product, matrix_vector_mul, outer_product,
                              zero_tangent_like)
from ._src.numpy_tools import create_diagonal_array, np_abs_square
from ._src.partial import Partial
from ._src.shims import custom_jvp, custom_jvp_method, custom_vjp, custom_vjp_method, jit
from ._src.testing import (assert_tree_allclose, get_relative_test_string, get_test_string,
                           tree_allclose)
from ._src.tree_tools import tree_map_with_path

__all__ = [
    'Array',
    'BooleanArray',
    'BooleanNumeric',
    'Complex',
    'ComplexArray',
    'ComplexNumeric',
    'Integral',
    'IntegralArray',
    'IntegralNumeric',
    'JaxAbstractClass',
    'JaxArray',
    'JaxBooleanArray',
    'JaxComplexArray',
    'JaxIntegralArray',
    'JaxRealArray',
    'KeyArray',
    'NumpyArray',
    'NumpyBooleanArray',
    'NumpyBooleanNumeric',
    'NumpyComplexArray',
    'NumpyComplexNumeric',
    'NumpyIntegralArray',
    'NumpyIntegralNumeric',
    'NumpyRealArray',
    'NumpyRealNumeric',
    'Partial',
    'PyTree',
    'Real',
    'RealArray',
    'RealNumeric',
    'Shape',
    'ShapeLike',
    'SliceLike',
    'TapFunctionTransforms',
    'abs_square',
    'abstract_custom_jvp',
    'abstract_jit',
    'assert_tree_allclose',
    'copy_cotangent',
    'cotangent_combinator',
    'create_diagonal_array',
    'custom_jvp',
    'custom_jvp_method',
    'custom_vjp',
    'custom_vjp_method',
    'dataclasses',
    'default_atol',
    'default_rtol',
    'default_tols',
    'diffused_leaky_integrate',
    'display_generic',
    'divide_nonnegative',
    'divide_where',
    'fixed_point',
    'get_relative_test_string',
    'get_test_string',
    'gradient',
    'graph_arrow',
    'graph_edge_name',
    'internal_print_generic',
    'inverse_softplus',
    'jit',
    'leaky_covariance',
    'leaky_data_weight',
    'leaky_integrate',
    'leaky_integrate_time_series',
    'matrix_dot_product',
    'matrix_vector_mul',
    'np_abs_square',
    'outer_product',
    'print_cotangent',
    'print_generic',
    'register_graph_as_jax_pytree',
    'register_graph_as_nnx_node',
    'replace_cotangent',
    'scale_cotangent',
    'tree_allclose',
    'tree_map_with_path',
    'zero_tangent_like',
]
