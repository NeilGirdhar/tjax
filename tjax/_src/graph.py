from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, MutableSet, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, override

from jax.tree_util import register_pytree_with_keys
from rich.tree import Tree

from .annotations import PyTree
from .display.display_generic import _verify, display_class, display_generic

__all__: list[str] = []


def graph_arrow(directed: bool) -> str:  # noqa: FBT001
    return '⟶' if directed else '↔'


def graph_edge_name(arrow: str, source: str, target: str) -> str:
    return f"{source}{arrow}{target}"


@dataclass(frozen=True)
class GraphNodeKey:
    """Struct for use with :func:`jax.tree_util.register_pytree_with_keys`."""
    key: Hashable

    @override
    def __str__(self) -> str:
        return f'.node[{self.key!r}]'

    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return (not isinstance(value, GraphNodeKey)
                or self.key < value.key)  # type: ignore # pyright: ignore


@dataclass(frozen=True)
class GraphEdgeKey:
    """Struct for use with :func:`jax.tree_util.register_pytree_with_keys`."""
    source: Hashable
    target: Hashable

    @override
    def __str__(self) -> str:
        return f'.edge[{self.source!r}, {self.target!r}]'

    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return isinstance(value, GraphEdgeKey) and (
                (self.source, self.target) < (value.source, value.target))


@dataclass(frozen=True)
class UndirectedGraphEdgeKey(GraphEdgeKey):
    @override
    def __hash__(self) -> int:
        return hash(self.source) ^ hash(self.target)

    @override
    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return isinstance(value, GraphEdgeKey) and (
                sorted((self.source, self.target))  # type: ignore # pyright: ignore
                < sorted((value.source, value.target)))  # type: ignore # pyright: ignore


GraphAuxData: TypeAlias = list[GraphNodeKey | GraphEdgeKey]  # The auxilliary data for Jax.
GraphData: TypeAlias = dict[str, Any]  # What networkx stores in the graph.


try:
    import networkx as nx
except ImportError:
    msg = "NetworkX not available"
    def register_graph_as_jax_pytree(graph_type: type[Any]) -> None:
        raise RuntimeError(msg)
    def register_graph_as_nnx_node(graph_type: type[Any]) -> None:
        raise RuntimeError(msg)
else:
    if TYPE_CHECKING:
        Graph: TypeAlias = nx.Graph[Any]
    else:
        Graph: TypeAlias = nx.Graph
    T = TypeVar('T', bound="nx.Graph[Any]")

    def nodes_helper(graph: nx.Graph[Any]) -> tuple[list[GraphNodeKey], list[PyTree]]:
        node_data = graph.nodes.data()
        node_keys = [GraphNodeKey(name) for name, _ in node_data]
        node_values = [value for _, value in node_data]
        return node_keys, node_values

    def edges_helper(graph: nx.Graph[Any]) -> tuple[list[GraphEdgeKey], list[PyTree]]:
        directed = isinstance(graph, nx.DiGraph)
        edge_data = graph.edges.data()
        edge_key: Callable[[Hashable, Hashable], GraphEdgeKey] = (
                GraphEdgeKey if directed else UndirectedGraphEdgeKey)
        edge_keys = [edge_key(source, target) for source, target, _ in edge_data]
        edge_data = [data for _, _, data in edge_data]
        return edge_keys, edge_data

    def flatten_tree(graph: nx.Graph[Any], /) -> tuple[list[GraphData], GraphAuxData]:
        node_keys, node_values = nodes_helper(graph)
        edge_keys, edge_values = edges_helper(graph)
        return node_values + edge_values, node_keys + edge_keys

    def register_graph_as_jax_pytree(graph_type: type[nx.Graph[Any]]) -> None:  # pyright: ignore
        def unflatten_tree(keys: GraphAuxData, values: Iterable[GraphData], /) -> nx.Graph[Any]:
            graph = graph_type()
            for key, value in zip(keys, values, strict=True):
                match key:
                    case GraphNodeKey(name):
                        graph.add_node(name, **value)
                    case GraphEdgeKey(source, target):
                        graph.add_edge(source, target, **value)
            return graph

        def flatten_with_keys(graph: nx.Graph[Any], /
                              ) -> tuple[Iterable[tuple[GraphNodeKey | GraphEdgeKey, GraphData]],
                                         GraphAuxData]:
            values, keys = flatten_tree(graph)
            return (zip(keys, values, strict=True), keys)

        flatten_with_keys_ = cast(
                Callable[[Graph], tuple[Iterable[tuple[Hashable, Any]], Hashable]],
                flatten_with_keys)
        unflatten_tree_ = cast(Callable[[Hashable, Any], Graph], unflatten_tree)
        flatten_tree_ = cast(Callable[[Graph], tuple[Iterable[Any], Hashable]],
                             flatten_tree)
        register_pytree_with_keys(graph_type, flatten_with_keys_, unflatten_tree_, flatten_tree_)

    @display_generic.register(nx.Graph)
    def _(value: nx.Graph[Any],
          *,
          seen: MutableSet[int] | None = None,
          key: str = '',
          ) -> Tree:
        if seen is None:
            seen = set()
        with _verify(value, seen, key) as x:
            if x:
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
        from flax.nnx.nnx.graph import register_graph_node_type
        from flax.typing import Key
    except ImportError:
        msg = "NNX not available"
        def register_graph_as_nnx_node(graph_type: type[Any]) -> None:
            raise RuntimeError(msg)
    else:
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

            register_graph_node_type(graph_type,
                                     flatten_graph,
                                     set_key_graph,
                                     pop_key_graph,
                                     create_empty_graph,
                                     clear_graph)
