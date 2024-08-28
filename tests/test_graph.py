from __future__ import annotations

from typing import Any

import pytest
from jax import jit, tree
from jax.tree_util import tree_flatten_with_path

from tjax import register_graph_as_jax_pytree

try:
    import networkx as nx
except ImportError:
    pytest.skip("Skipping graph test", allow_module_level=True)
else:
    register_graph_as_jax_pytree(nx.DiGraph)


@pytest.fixture(scope='session')
def graph() -> nx.DiGraph[Any]:
    g = nx.DiGraph()
    g.add_node('a', y=2.0)
    g.add_node('b', z=3.0)
    g.add_node('c', w=4.0)
    g.add_edge('a', 'b', x=5.0)
    g.add_edge('c', 'b', x=7.0)
    return g


@jit
def f(x: nx.DiGraph[Any]) -> nx.DiGraph[Any]:
    return x


def test_rebuild(graph: nx.DiGraph[Any]) -> None:
    values, tree_def = tree.flatten(graph)
    rebuilt_graph = tree.unflatten(tree_def, values)
    assert nx.utils.graphs_equal(graph, rebuilt_graph)


def test_rebuild_jit(graph: nx.DiGraph[Any]) -> None:
    rebuilt_graph = f(graph)
    assert nx.utils.graphs_equal(graph, rebuilt_graph)


def test_flatten_flavors(graph: nx.DiGraph[Any]) -> None:
    values_a, tree_def_a = tree.flatten(graph)
    keys_and_values, tree_def_b = tree_flatten_with_path(graph)
    key_paths, values_b = zip(*keys_and_values, strict=True)
    assert hash(tree_def_a) == hash(tree_def_b)
    assert values_a == list(values_b)
    assert values_a == [2.0, 3.0, 4.0, 5.0, 7.0]
    simplified = {"".join(str(x) for x in key_path)
                  for key_path in key_paths}
    assert simplified == {".node['a']['y']",
                          ".node['b']['z']",
                          ".node['c']['w']",
                          ".edge['a', 'b']['x']",
                          ".edge['c', 'b']['x']"}
