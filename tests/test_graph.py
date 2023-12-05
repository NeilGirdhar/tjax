import pytest
from jax.tree_util import tree_flatten_with_path

import tjax  # noqa: F401

try:
    import networkx as nx
except ImportError:
    pytest.skip("Skipping nnx graph test", allow_module_level=True)


def test_graph_for_jax() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b', x=1.0)
    # tuple[list[tuple[KeyPath, Any]], PyTreeDef]
    ((keypaths, value),), _ = tree_flatten_with_path(g)
    assert value == 1.0  # noqa: PLR2004
    assert len(keypaths) == 3  # noqa: PLR2004
    keys = [keypath.key for keypath in keypaths]
    assert keys == [1, ('a', 'b'), 'x']


def test_graph_for_flax() -> None:
    try:
        from flax.experimental import nnx  # noqa: PLC0415
    except ImportError:
        pytest.skip("Skipping nnx graph test")
    g = nx.DiGraph()
    v = nnx.Variable(1.0)
    g.add_edge('a', 'b', x=v)
    state, _ = nnx.graph_utils.graph_flatten(g)  # pyright: ignore
    assert state['a⟶b']['x'] == 1.0  # noqa: PLR2004
