from __future__ import annotations

from collections.abc import MutableSet
from typing import Any

from rich.tree import Tree

from ..display.display_generic import _verify, display_class, display_generic


def graph_arrow(directed: bool) -> str:  # noqa: FBT001
    return '⟶' if directed else '↔'


def graph_edge_name(arrow: str, source: str, target: str) -> str:
    return f"{source}{arrow}{target}"


try:
    import networkx as nx
except ImportError:
    pass
else:
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
