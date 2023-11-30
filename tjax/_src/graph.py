from __future__ import annotations

from collections.abc import Callable, Hashable, MutableSet, Sequence
from typing import Any, TypeVar

from jax.tree_util import register_pytree_node
from rich.tree import Tree

from .annotations import PyTree
from .display import BatchDimensionIterator, BatchDimensions, display_class, display_generic

__all__: list[str] = []


try:
    import networkx as nx
except ImportError:
    pass
else:
    T = TypeVar('T', bound=nx.Graph)

    def register_graph_as_jax_pytree(cls: type[T]) -> None:
        def tree_unflatten(hashed: Hashable, trees: Sequence[PyTree]) -> T:
            node_dicts, edge_dicts = trees

            if not isinstance(node_dicts, dict):
                raise TypeError
            if not isinstance(edge_dicts, dict):
                raise TypeError

            graph = cls()
            graph.add_nodes_from(node_dicts.items())
            graph.add_edges_from([(source, target, data)
                                  for (source, target), data in edge_dicts.items()])

            return graph

        def tree_flatten(graph: T) -> tuple[Sequence[PyTree], Hashable]:
            return ((dict(graph.nodes), dict(graph.edges)), None)

        register_pytree_node(cls, tree_flatten, tree_unflatten)

    register_graph_as_jax_pytree(nx.Graph)
    register_graph_as_jax_pytree(nx.DiGraph)

    try:
        from flax.experimental.nnx.nnx.graph_utils import register_node_type
    except ImportError:
        pass
    else:
        def _flatten_graph(node: nx.Graph
                           ) -> tuple[
                                   tuple[tuple[str, Any], ...],
                                   None]:
            return ((('nodes', tuple(node.nodes.data())), ('edges', tuple(node.edges.data()))),
                    None)

        def _get_key_graph(node: nx.Graph, key: str) -> Any:
            return node[key]

        def _set_key_graph(node: nx.Graph, key: str, value: Any) -> nx.Graph:
            msg = f"'{type(node).__name__}' object is immutable; it does not support assignment."
            raise ValueError(msg)

        def _has_key_graph(node: nx.Graph, key: str) -> bool:
            return key in {'nodes', 'edges'}

        def _all_keys_graph(node: nx.Graph) -> tuple[str, ...]:
            return ('nodes', 'edges')

        def closure_create_empty(node_type: type[T]) -> Callable[[None], T]:
            def _create_empty_graph(metadata: None) -> T:
                return node_type()
            return _create_empty_graph

        def _init_graph(node: nx.Graph,
                        items: tuple[tuple[str, Any], ...]
                        ) -> None:
            node.update(**dict(items))

        for node_type in [nx.Graph, nx.DiGraph]:
            register_node_type(node_type,
                               _flatten_graph,
                               _get_key_graph,
                               _set_key_graph,
                               _has_key_graph,
                               _all_keys_graph,
                               create_empty=closure_create_empty(node_type),
                               init=_init_graph)

    @display_generic.register
    def _(value: nx.Graph,
          *,
          seen: MutableSet[int],
          show_values: bool = True,
          key: str = '',
          batch_dims: BatchDimensions | None = None) -> Tree:
        directed = isinstance(value, nx.DiGraph)
        arrow = '⟶' if directed else '↔'
        retval = display_class(key, type(value))
        bdi = BatchDimensionIterator(batch_dims)
        for name, node in value.nodes.items():
            sub_batch_dims = bdi.advance(node)
            retval.children.append(display_generic(node, seen=seen, show_values=show_values,
                                                   key=name, batch_dims=sub_batch_dims))
        for (source, target), edge in value.edges.items():
            key = f"{source}{arrow}{target}"
            sub_batch_dims = bdi.advance(edge)
            retval.children.append(display_generic(edge, seen=seen, show_values=show_values,
                                                   key=key, batch_dims=sub_batch_dims))
        return retval
