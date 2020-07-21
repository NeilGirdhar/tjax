from typing import Hashable, List, Sequence, Tuple, Type, TypeVar

import networkx as nx
from jax.tree_util import register_pytree_node

from .annotations import PyTree

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
        graph.add_edges_from([(source, target, data)  # type: ignore
                              for (source, target), data in edge_dicts.items()])

        return graph

    def tree_flatten(self: T) -> Tuple[Sequence[PyTree], Hashable]:
        # https://github.com/python/mypy/issues/8768
        return ((dict(self.nodes), dict(self.edges)), None)  # type: ignore

    register_pytree_node(cls, tree_flatten, tree_unflatten)


register_graph_as_jax_pytree(nx.Graph)
register_graph_as_jax_pytree(nx.DiGraph)
