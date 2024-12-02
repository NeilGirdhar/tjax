from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from jax.tree_util import register_pytree_with_keys

from ..annotations import PyTree
from .types_ import GraphAuxData, GraphData, GraphEdgeKey, GraphNodeKey, UndirectedGraphEdgeKey

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
                "Callable[[Graph], tuple[Iterable[tuple[Hashable, Any]], Hashable]]",
                flatten_with_keys)
        unflatten_tree_ = cast("Callable[[Hashable, Any], Graph]", unflatten_tree)
        flatten_tree_ = cast("Callable[[Graph], tuple[Iterable[Any], Hashable]]",
                             flatten_tree)
        register_pytree_with_keys(graph_type, flatten_with_keys_, unflatten_tree_, flatten_tree_)
