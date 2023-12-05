import pytest
import tjax  # noqa: F401

try:
    import networkx as nx
    from flax.experimental import nnx
except ImportError:
    pytest.skip("Skipping nnx graph test", allow_module_level=True)


def test_graph() -> None:
    g = nx.DiGraph()
    v = nnx.Variable(1.0)
    g.add_edge('a', 'b', x=v)
    state, _ = nnx.graph_utils.graph_flatten(g)  # pyright: ignore
    assert state['a⟶b']['x'] == 1.0  # noqa: PLR2004
