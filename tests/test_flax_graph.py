from __future__ import annotations

from typing import Any

import pytest

from tjax import register_graph_as_nnx_node

try:
    import networkx as nx
    from flax.experimental import nnx
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
    state, graph_def = nnx.graph_utils.graph_flatten(graph)
    rebuilt_graph = nnx.graph_utils.graph_unflatten(graph_def, state)
    assert nx.utils.graphs_equal(graph, rebuilt_graph)


def test_clone(graph: nx.DiGraph[Any]) -> None:
    class Model(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph = graph

    m = Model()
    n = m.clone()
    assert nx.utils.graphs_equal(m.graph, n.graph)


def test_flatten(graph: nx.DiGraph[Any]) -> None:
    state, _ = nnx.graph_utils.graph_flatten(graph)
    assert state['a⟶b']['x'] == 4.0  # noqa: PLR2004