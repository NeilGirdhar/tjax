from __future__ import annotations

from functools import WRAPPER_ASSIGNMENTS

__all__: list[str] = []


abstract_jit_marker = '_abstract_jit'
abstract_custom_jvp_marker = '_abstract_custom_jvp'
all_wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__isabstractmethod__', '__override__',
                           abstract_jit_marker, abstract_custom_jvp_marker)
