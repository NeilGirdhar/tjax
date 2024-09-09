"""Tools for the differential programming library [JAX](https://github.com/google/jax)."""
from . import dataclasses, gradient
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
                                   replace_cotangent, scale_cotangent, zero_tangent_like)
from ._src.display.display_generic import display_generic
from ._src.display.internal import internal_print_generic
from ._src.display.print_generic import print_generic
from ._src.dtype_tools import cast_to_result_type, result_type
from ._src.dtypes import default_atol, default_rtol, default_tols
from ._src.graph.display import graph_arrow, graph_edge_name
from ._src.graph.register_flax import register_graph_as_nnx_node
from ._src.graph.register_jax import register_graph_as_jax_pytree
from ._src.graph.types_ import GraphEdgeKey, GraphNodeKey, UndirectedGraphEdgeKey
from ._src.leaky_integral import (diffused_leaky_integrate, leaky_covariance, leaky_data_weight,
                                  leaky_integrate, leaky_integrate_time_series)
from ._src.math_tools import (abs_square, create_diagonal_array, divide_nonnegative, divide_where,
                              inverse_softplus, matrix_dot_product, matrix_vector_mul,
                              outer_product, softplus)
from ._src.partial import Partial
from ._src.rng import RngStream, create_streams, fork_streams, sample_streams
from ._src.shims import custom_jvp, custom_jvp_method, custom_vjp, custom_vjp_method, hessian, jit
from ._src.testing import (assert_tree_allclose, get_relative_test_string, get_test_string,
                           tree_allclose)
from ._src.tree_tools import dynamic_tree_all

__all__ = [
    'Array',
    'BooleanArray',
    'BooleanNumeric',
    'Complex',
    'ComplexArray',
    'ComplexNumeric',
    'GraphEdgeKey',
    'GraphNodeKey',
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
    'RngStream',
    'Shape',
    'ShapeLike',
    'SliceLike',
    'TapFunctionTransforms',
    'UndirectedGraphEdgeKey',
    'abs_square',
    'abstract_custom_jvp',
    'abstract_jit',
    'assert_tree_allclose',
    'cast_to_result_type',
    'copy_cotangent',
    'cotangent_combinator',
    'create_diagonal_array',
    'create_streams',
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
    'dynamic_tree_all',
    'fork_streams',
    'get_relative_test_string',
    'get_test_string',
    'gradient',
    'graph_arrow',
    'graph_edge_name',
    'hessian',
    'internal_print_generic',
    'inverse_softplus',
    'jit',
    'leaky_covariance',
    'leaky_data_weight',
    'leaky_integrate',
    'leaky_integrate_time_series',
    'matrix_dot_product',
    'matrix_vector_mul',
    'outer_product',
    'print_cotangent',
    'print_generic',
    'register_graph_as_jax_pytree',
    'register_graph_as_nnx_node',
    'replace_cotangent',
    'result_type',
    'sample_streams',
    'scale_cotangent',
    'softplus',
    'tree_allclose',
    'zero_tangent_like',
]
