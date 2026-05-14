from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from _typeshed import SupportsRichComparison


@dataclass(frozen=True)
class GraphNodeKey:
    """Struct for use with :func:`jax.tree_util.register_pytree_with_keys`.

    Ordering assumes node keys are mutually comparable. Heterogeneous node-key types may therefore
    fail when an ordered traversal is requested.
    """

    key: SupportsRichComparison

    @override
    def __str__(self) -> str:
        return f".node[{self.key!r}]"

    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return not isinstance(value, GraphNodeKey) or self.key < value.key  # type: ignore


@dataclass(frozen=True)
class GraphEdgeKey:
    """Struct for use with :func:`jax.tree_util.register_pytree_with_keys`.

    Ordering assumes source and target keys are mutually comparable. Heterogeneous endpoint types
    may therefore fail when an ordered traversal is requested.
    """

    source: SupportsRichComparison
    target: SupportsRichComparison

    @override
    def __str__(self) -> str:
        return f".edge[{self.source!r}, {self.target!r}]"

    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return isinstance(value, GraphEdgeKey) and (
            (self.source, self.target) < (value.source, value.target)  # type: ignore
        )


@dataclass(frozen=True)
class UndirectedGraphEdgeKey(GraphEdgeKey):
    """An undirected graph edge key with the same comparability assumptions as GraphEdgeKey."""

    @override
    def __hash__(self) -> int:
        return hash(self.source) ^ hash(self.target)

    @override
    def __lt__(self, value: GraphNodeKey | GraphEdgeKey, /) -> bool:
        return isinstance(value, GraphEdgeKey) and (
            sorted((self.source, self.target)) < sorted((value.source, value.target))
        )


type GraphAuxData = list[GraphNodeKey | GraphEdgeKey]  # The auxilliary data for Jax.
type GraphData = dict[str, Any]  # What networkx stores in the graph.
