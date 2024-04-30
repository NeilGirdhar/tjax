from __future__ import annotations

from collections.abc import Generator, Iterable, MutableSet
from typing import Any, TypeVar

from jax.tree_util import register_pytree_with_keys
from rich.tree import Tree

from .annotations import PyTree
from .display.display_generic import _verify, display_class, display_generic

__all__: list[str] = []


def graph_arrow(directed: bool) -> str:  # noqa: FBT001
    return '⟶' if directed else '↔'


def graph_edge_name(arrow: str, source: str, target: str) -> str:
    return f"{source}{arrow}{target}"


try:
    import networkx as nx
except ImportError:
    msg = "NetworkX not available"
    def register_graph_as_jax_pytree(graph_type: type[Any]) -> None:
        raise RuntimeError(msg)
    def register_graph_as_nnx_node(graph_type: type[Any]) -> None:
        raise RuntimeError(msg)
else:
    T = TypeVar('T', bound="nx.Graph[Any]")

    def validate_node_name(name: Any,
                           arrow: str
                           ) -> None:
        if not isinstance(name, str):
            raise TypeError
        if arrow in name:
            raise ValueError

    def flatten_helper(graph: nx.Graph[Any],
                       arrow: str
                       ) -> Generator[tuple[str, Any], None, None]:
        for name, data in sorted(graph.nodes.data()):
            validate_node_name(name, arrow)
            yield name, data

        def undirected_edge_key(source_target_data: tuple[str, str, Any]) -> tuple[str, ...]:
            source, target, _ = source_target_data
            return tuple(sorted([source, target]))
        edge_data = graph.edges.data()
        directed = isinstance(graph, nx.DiGraph)
        if not directed:
            edge_data = sorted(edge_data, key=undirected_edge_key)
        for source, target, data in edge_data:
            validate_node_name(source, arrow)
            validate_node_name(target, arrow)
            new_source, new_target = ((source, target)
                                      if directed
                                      else sorted([source, target]))
            yield graph_edge_name(arrow, new_source, new_target), data

    def init_graph(graph: nx.Graph[Any],
                   items: Iterable[tuple[str, Any]],
                   arrow: str
                   ) -> None:
        for key, value in items:
            if not isinstance(value, dict):
                raise TypeError
            if arrow in key:
                source, target = key.split(arrow, 2)
                graph.add_edge(source, target, **value)
            else:
                graph.add_node(key, **value)

    def register_graph_as_jax_pytree(graph_type: type[nx.Graph[Any]]) -> None:  # pyright: ignore
        arrow = graph_arrow(issubclass(graph_type, nx.DiGraph))

        def unflatten_tree(names: tuple[str, ...], values: Iterable[Any], /) -> nx.Graph[Any]:
            graph = graph_type()
            init_graph(graph, zip(names, values, strict=True), arrow)
            return graph

        def flatten_with_keys(graph: nx.Graph[Any], /
                              ) -> tuple[Iterable[tuple[str, Any]], tuple[str, ...]]:
            names_and_values = tuple(flatten_helper(graph, arrow))
            names = tuple(name for name, _ in names_and_values)
            return (names_and_values, names)

        def flatten_tree(graph: nx.Graph[Any], /) -> tuple[Iterable[PyTree], tuple[str, ...]]:
            names_and_values = tuple(flatten_helper(graph, arrow))
            names, values = zip(*names_and_values, strict=True)
            return values, names

        register_pytree_with_keys(graph_type, flatten_with_keys, unflatten_tree, flatten_tree)

    @display_generic.register(nx.Graph)
    def _(value: nx.Graph[Any],
          *,
          seen: MutableSet[int] | None = None,
          key: str = '',
          ) -> Tree:
        if seen is None:
            seen = set()
        if (x := _verify(value, seen, key)) is not None:
            return x
        arrow = graph_arrow(isinstance(value, nx.DiGraph))
        retval = display_class(key, type(value))
        for name, node in value.nodes.items():
            retval.children.append(display_generic(node, seen=seen, key=name))
        for (source, target), edge in value.edges.items():
            key = graph_edge_name(arrow, source, target)
            retval.children.append(display_generic(edge, seen=seen, key=key))
        return retval

    try:
        from flax.experimental.nnx.nnx.graph import register_graph_node_type
        from flax.typing import Key
    except ImportError:
        msg = "NNX not available"
        def register_graph_as_nnx_node(graph_type: type[Any]) -> None:
            raise RuntimeError(msg)
    else:
        def register_graph_as_nnx_node(graph_type: type[T]) -> None:  # pyright: ignore
            arrow = graph_arrow(issubclass(graph_type, nx.DiGraph))

            # flatten: Callable[[Node], tuple[Sequence[tuple[str, Leaf]], AuxData]],
            def flatten_graph(graph: T, /) -> tuple[tuple[tuple[str, Any], ...], None]:
                t = tuple(flatten_helper(graph, arrow))
                return t, None

            # set_key: Callable[[Node, Key, Leaf], None],
            def set_key_graph(graph: T, key: Key, value: Any, /) -> None:
                if not isinstance(value, dict):
                    raise TypeError
                assert isinstance(key, str)
                d: dict[str, Any]
                if arrow in key:
                    source, target = key.split(arrow, 2)
                    if not graph.has_edge(source, target):
                        graph.add_edge(source, target, **value)
                        return
                    d = graph.edges[source, target]
                elif not graph.has_node(key):
                    graph.add_node(key, **value)
                    return
                else:
                    d = graph.nodes[key]
                d.clear()
                d.update(value)

            # pop_key: Callable[[Node, Key], Leaf],
            def pop_key_graph(graph: T, key: Key, /) -> Any:
                assert isinstance(key, str)
                if arrow in key:
                    source, target = key.split(arrow, 2)
                    retval = graph.edges[source, target]
                    graph.remove_edge(source, target)
                    return retval
                retval = graph.nodes[key]
                graph.remove_node(key)
                return retval

            # create_empty: Callable[[AuxData], Node],
            def create_empty_graph(metadata: None, /) -> T:
                del metadata
                return graph_type()

            # clear: Callable[[Node, AuxData], None],
            def clear_graph(graph: T, metadata: None, /) -> None:
                del metadata
                graph.clear()

            register_graph_node_type(graph_type,
                                     flatten_graph,
                                     set_key_graph,
                                     pop_key_graph,
                                     create_empty_graph,
                                     clear_graph)
