"""Tools for the differential programming library [JAX](https://github.com/google/jax)."""
from . import dataclasses, gradient
from ._src.abstract_method_decorators import JaxAbstractClass, abstract_custom_jvp, abstract_jit
from ._src.annotations import (Array, BooleanArray, BooleanNumeric, ComplexArray, ComplexNumeric,
                               IntegralArray, IntegralNumeric, JaxArray, JaxBooleanArray,
                               JaxComplexArray, JaxIntegralArray, JaxRealArray, KeyArray,
                               NumpyArray, NumpyBooleanArray, NumpyBooleanNumeric,
                               NumpyComplexArray, NumpyComplexNumeric, NumpyIntegralArray,
                               NumpyIntegralNumeric, NumpyRealArray, NumpyRealNumeric, PyTree,
                               RealArray, RealNumeric, Shape, ShapeLike, SliceLike)
from ._src.cotangent_tools import (copy_cotangent, cotangent_combinator, print_cotangent,
                                   replace_cotangent, reverse_scale_cotangent, scale_cotangent,
                                   zero_from_primal)
from ._src.display.display_generic import display_generic
from ._src.display.internal import internal_print_generic
from ._src.display.print_generic import print_generic
from ._src.dtype_tools import cast_to_result_type
from ._src.graph.display import graph_arrow, graph_edge_name
from ._src.graph.register_flax import register_graph_as_nnx_node
from ._src.graph.register_jax import register_graph_as_jax_pytree
from ._src.graph.types_ import GraphEdgeKey, GraphNodeKey, UndirectedGraphEdgeKey
from ._src.k_means import cluster_with_k_means
from ._src.leaky_integral import diffused_leaky_integrate, leaky_data_weight, leaky_integrate
from ._src.math_tools import (abs_square, divide_where, inverse_softplus, log_softplus,
                              matrix_dot_product, matrix_vector_mul, normalize, outer_product,
                              softplus, stop_gradient, sublinear_softplus)
from ._src.partial import Partial
from ._src.rng import RngStream, create_streams, fork_streams, sample_streams
from ._src.shims import custom_jvp, custom_jvp_method, custom_vjp, custom_vjp_method, hessian, jit
from ._src.testing import (assert_tree_allclose, get_relative_test_string, get_test_string,
                           tree_allclose)
from ._src.tree_tools import dynamic_tree_all, element_count, tree_sum

__all__ = [
    'Array',
    'BooleanArray',
    'BooleanNumeric',
    'ComplexArray',
    'ComplexNumeric',
    'GraphEdgeKey',
    'GraphNodeKey',
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
    'RealArray',
    'RealNumeric',
    'RngStream',
    'Shape',
    'ShapeLike',
    'SliceLike',
    'UndirectedGraphEdgeKey',
    'abs_square',
    'abstract_custom_jvp',
    'abstract_jit',
    'assert_tree_allclose',
    'cast_to_result_type',
    'cluster_with_k_means',
    'copy_cotangent',
    'cotangent_combinator',
    'create_streams',
    'custom_jvp',
    'custom_jvp_method',
    'custom_vjp',
    'custom_vjp_method',
    'dataclasses',
    'diffused_leaky_integrate',
    'display_generic',
    'divide_where',
    'dynamic_tree_all',
    'element_count',
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
    'leaky_data_weight',
    'leaky_integrate',
    'log_softplus',
    'matrix_dot_product',
    'matrix_vector_mul',
    'normalize',
    'outer_product',
    'print_cotangent',
    'print_generic',
    'register_graph_as_jax_pytree',
    'register_graph_as_nnx_node',
    'replace_cotangent',
    'reverse_scale_cotangent',
    'sample_streams',
    'scale_cotangent',
    'softplus',
    'stop_gradient',
    'sublinear_softplus',
    'tree_allclose',
    'tree_sum',
    'zero_from_primal',
]
