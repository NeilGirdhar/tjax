from __future__ import annotations

from typing import Any

import pytest

from tjax import GraphEdgeKey, register_graph_as_nnx_node

try:
    import networkx as nx
    from flax import nnx
except ImportError:
    pytest.skip("Skipping NNX graph test", allow_module_level=True)
else:
    register_graph_as_nnx_node(nx.DiGraph)


@pytest.fixture(scope='session')
def graph() -> nx.DiGraph[Any]:
    v = nnx.Variable(2.0)
    w = nnx.Variable(3.0)
    x = nnx.Variable(4.0)
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
    substate = state[GraphEdgeKey('a', 'b')]
    assert isinstance(substate, nnx.State)
    variable = substate['x']
    assert isinstance(variable, nnx.VariableState)
    assert variable.value == 4.0  # noqa: PLR2004
