from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableSet
from typing import Any, Dict, Hashable, Sequence, TypeVar

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
        from flax.serialization import from_state_dict, register_serialization_state, to_state_dict
    except ImportError:
        pass
    else:
        def register_graph_as_flax_state_dict(cls: type[T]) -> None:
            def ty_to_state_dict(graph: T) -> dict[str, Any]:
                edge_dict_of_dicts = defaultdict[Any, Dict[Any, Any]](dict)
                for (source, target), edge_dict in dict(graph.edges).items():
                    edge_dict_of_dicts[source][target] = edge_dict
                return {'nodes': to_state_dict(dict(graph.nodes)),
                        'edges': to_state_dict(dict(edge_dict_of_dicts))}

            def ty_from_state_dict(graph: T, state_dict: dict[str, Any]) -> T:
                retval = type(graph)()
                for node_name, node_dict in state_dict['nodes'].items():
                    retval.add_node(node_name, **from_state_dict(graph.nodes[node_name], node_dict))
                for source, target_and_edge_dict in state_dict['edges'].items():
                    for target, edge_dict in target_and_edge_dict.items():
                        retval.add_edge(source, target,
                                        **from_state_dict(graph.edges[source, target], edge_dict))
                return retval

            register_serialization_state(cls, ty_to_state_dict,  # type: ignore[no-untyped-call]
                                         ty_from_state_dict)

        register_graph_as_flax_state_dict(nx.Graph)
        register_graph_as_flax_state_dict(nx.DiGraph)

    @display_generic.register
    def _(value: nx.Graph,
          *,
          seen: MutableSet[int],
          show_values: bool = True,
          key: str = '',
          batch_dims: BatchDimensions | None = None) -> Tree:
        directed = isinstance(value, nx.DiGraph)
        arrow = '⟶  ' if directed else '🡘 '
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
