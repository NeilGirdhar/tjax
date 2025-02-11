from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

from .types_ import GraphEdgeKey, GraphNodeKey

T = TypeVar('T', bound="nx.Graph[Any]")


try:
    import networkx as nx
    from flax.nnx.graph import register_graph_node_type
except ImportError:
    msg = "NetworkX or NNX not available"
    def register_graph_as_nnx_node(graph_type: type[Any]) -> None:
        raise RuntimeError(msg)
else:
    from flax.typing import Key

    from .register_jax import flatten_tree

    def register_graph_as_nnx_node(graph_type: type[T]) -> None:  # pyright: ignore
        # flatten: Callable[[Node], tuple[Sequence[tuple[Key, Leaf]], AuxData]],
        def flatten_graph(graph: T, /
                          ) -> tuple[Sequence[tuple[Key, Any]], None]:
            values, keys = flatten_tree(graph)
            return list(zip(keys, values, strict=True)), None

        # set_key: Callable[[Node, Key, Leaf], None],
        def set_key_graph(graph: T, key: Key, value: Any, /) -> None:
            if not isinstance(value, dict):
                raise TypeError
            d: dict[str, Any]
            match key:
                case GraphNodeKey(name):
                    if not graph.has_node(name):
                        graph.add_node(name, **value)
                        return
                    d = graph.nodes[name]
                case GraphEdgeKey(source, target):
                    if not graph.has_edge(source, target):
                        graph.add_edge(source, target, **value)
                        return
                    d = graph.edges[source, target]
                case _:
                    raise TypeError
            d.clear()
            d.update(value)

        # pop_key: Callable[[Node, Key], Leaf],
        def pop_key_graph(graph: T, key: Key, /) -> Any:
            match key:
                case GraphNodeKey(name):
                    retval = graph.nodes[name]
                    graph.remove_node(name)
                    return retval
                case GraphEdgeKey(source, target):
                    retval = graph.edges[source, target]
                    graph.remove_edge(source, target)
                    return retval
                case _:
                    raise TypeError

        # create_empty: Callable[[AuxData], Node],
        def create_empty_graph(metadata: None, /) -> T:
            del metadata
            return graph_type()

        # clear: Callable[[Node, AuxData], None],
        def clear_graph(graph: T, /) -> None:
            graph.clear()

        # init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None],
        def init_graph(graph: T, it: Iterable[tuple[Key, Any]]) -> None:
            for key, value in it:
                set_key_graph(graph, key, value)

        register_graph_node_type(graph_type,
                                 flatten_graph,
                                 set_key_graph,
                                 pop_key_graph,
                                 create_empty_graph,
                                 clear_graph,
                                 init_graph)
