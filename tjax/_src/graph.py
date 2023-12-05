from __future__ import annotations

from collections.abc import Callable, Generator, Hashable, MutableSet, Sequence
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
        arrow = '⟶' if True else '↔'
        def validate(name: Any) -> None:
            if not isinstance(name, str):
                raise TypeError
            if arrow in name:
                raise ValueError

        def flatten_helper(node: nx.Graph
                           ) -> Generator[tuple[str, Any], None, None]:
            for name, data in node.nodes.data():
                validate(name)
                yield name, data
            for source, target, data in node.edges.data():  # pyright: ignore
                validate(source)
                validate(target)
                yield f"{source}{arrow}{target}", data

        def flatten_graph(node: nx.Graph
                          ) -> tuple[tuple[tuple[str, Any], ...], None]:
            t = tuple(flatten_helper(node))
            return t, None

        def get_key_graph(node: nx.Graph, key: str) -> Any:
            if arrow in key:
                source, target = key.split(arrow, 2)
                return node.edges[source, target]
            return node.nodes[key]

        def set_key_graph(node: nx.Graph, key: str, value: Any) -> nx.Graph:
            if not isinstance(value, dict):
                raise TypeError
            if arrow in key:
                source, target = key.split(arrow, 2)
                d = node.edges[source, target]
            else:
                d = node.nodes[key]
            d.clear()
            d.update(value)
            return d

        def has_key_graph(node: nx.Graph, key: str) -> bool:
            if arrow in key:
                source, target = key.split(arrow, 2)
                # Care here to return true for only ordered edges.
                return (source, target) in list(node.edges)
            return key in node

        def all_keys_helper(node: nx.Graph
                            ) -> Generator[str, None, None]:
            for name in node.nodes:
                validate(name)
                yield name
            for source, target in node.edges:
                validate(source)
                validate(target)
                yield f"{source}{arrow}{target}"

        def all_keys_graph(node: nx.Graph) -> tuple[str, ...]:
            return tuple(all_keys_helper(node))

        def closure_create_empty(node_type: type[T]) -> Callable[[None], T]:
            def create_empty_graph(metadata: None) -> T:
                return node_type()
            return create_empty_graph

        def init_graph(node: nx.Graph,
                       items: tuple[tuple[str, Any], ...]
                       ) -> None:
            for key, value in items:
                if not isinstance(value, dict):
                    raise TypeError
                if arrow in key:
                    source, target = key.split(arrow, 2)
                    node.add_edge(source, target, **value)
                else:
                    node.add_node(key, **value)

        for node_type in [nx.Graph, nx.DiGraph]:
            register_node_type(node_type,
                               flatten_graph,
                               get_key_graph,
                               set_key_graph,
                               has_key_graph,
                               all_keys_graph,
                               create_empty=closure_create_empty(node_type),
                               init=init_graph)

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
