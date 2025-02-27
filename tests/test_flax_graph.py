from __future__ import annotations

from typing import Any, TypeVar, override

import pytest

from tjax import GraphEdgeKey, register_graph_as_nnx_node

try:
    import networkx as nx
    from flax import nnx
except ImportError:
    pytest.skip("Skipping NNX graph test", allow_module_level=True)
else:
    register_graph_as_nnx_node(nx.DiGraph)


T = TypeVar('T')


class ComparableVariable(nnx.Variable[T]):
    @override
    def __hash__(self) -> int:
        raise NotImplementedError

    @override
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, ComparableVariable)
        return self.raw_value == other.raw_value


@pytest.fixture(scope='session')
def graph() -> nx.DiGraph[Any]:
    v = ComparableVariable(2.0)
    w = ComparableVariable(3.0)
    x = ComparableVariable(4.0)
    g = nx.DiGraph()
    g.add_node('a', y=v)
    g.add_node('b', z=w)
    g.add_edge('a', 'b', x=x)
    g.add_edge('c', 'b', x=x)
    return g


def test_rebuild(graph: nx.DiGraph[Any]) -> None:
    graph_def, state = nnx.graph.flatten(graph)
    rebuilt_graph = nnx.graph.unflatten(graph_def, state)
    assert nx.utils.graphs_equal(graph, rebuilt_graph)


def test_clone(graph: nx.DiGraph[Any]) -> None:
    class Model(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph = graph

    m = Model()
    n = nnx.graph.clone(m)
    assert nx.utils.graphs_equal(m.graph, n.graph)


def test_flatten(graph: nx.DiGraph[Any]) -> None:
    _, state = nnx.graph.flatten(graph)
    state_dict = dict(state)
    variable = state_dict[GraphEdgeKey('a', 'b'), 'x']
    assert isinstance(variable, nnx.VariableState)
    assert variable.value == 4.0  # noqa: PLR2004
