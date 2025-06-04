from __future__ import annotations

from typing import override

from rich.console import Console

from .print_generic import print_generic


class GenericString:
    """An object that can be passed to a logging function to lazily log using print_generic."""
    def __init__(self,
                 *args: object,
                 immediate: bool = True,
                 raise_on_nan: bool = True,
                 as_leaves: bool = False,
                 ) -> None:
        super().__init__()
        self.args = args
        self.raise_on_nan = raise_on_nan
        self.immediate = immediate
        self.as_leaves = as_leaves

    @override
    def __str__(self) -> str:
        console = Console(no_color=True)
        with console.capture() as capture:
            print_generic(*self.args, raise_on_nan=self.raise_on_nan, immediate=self.immediate,
                          as_leaves=self.as_leaves, console=console)
        return capture.get()
