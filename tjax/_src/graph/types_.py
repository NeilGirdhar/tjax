from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, TypeAlias, override


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
