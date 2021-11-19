from __future__ import annotations

from typing import Hashable, List, Optional, Sequence, Tuple, Type, TypeVar

import colorful as cf
import networkx as nx
from jax.tree_util import register_pytree_node

from .annotations import PyTree
from .display import BatchDimensionIterator, display_class, display_generic, display_key_and_value

__all__: List[str] = []

T = TypeVar('T', bound=nx.Graph)


def register_graph_as_jax_pytree(cls: Type[T]) -> None:
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

    def tree_flatten(self: T) -> Tuple[Sequence[PyTree], Hashable]:
        return ((dict(self.nodes), dict(self.edges)), None)

    register_pytree_node(cls, tree_flatten, tree_unflatten)


register_graph_as_jax_pytree(nx.Graph)
register_graph_as_jax_pytree(nx.DiGraph)


@display_generic.register
def _(value: nx.Graph,
      show_values: bool,
      indent: int = 0,
      batch_dims: Optional[Tuple[Optional[int], ...]] = None) -> str:
    directed = isinstance(value, nx.DiGraph)
    arrow = cf.base00('⟶  ' if directed else '🡘 ')
    retval = display_class(type(value))
    bdi = BatchDimensionIterator(batch_dims)
    for name, node in value.nodes.items():
        sub_batch_dims = bdi.advance(node)
        retval += display_key_and_value(name, node, ": ", show_values, indent + 1, sub_batch_dims)
    for (source, target), edge in value.edges.items():
        key = f"{source}{arrow}{target}"
        sub_batch_dims = bdi.advance(edge)
        retval += display_key_and_value(key, edge, ": ", show_values, indent, sub_batch_dims)
    return retval
