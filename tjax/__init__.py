"""
This library implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).
"""
from .color_stub import *
from .annotations import *
from .dataclass import *
from .display import *
from .dtypes import *
from .generator import *
from .graph import *
from .pytree_like import *
from .shims import *
from .testing import *
from .tools import *


__pdoc__ = {}
__pdoc__['real_dtype'] = False
__pdoc__['complex_dtype'] = False
__pdoc__['PyTreeLike'] = False
from .dataclass import document_dataclass
document_dataclass(__pdoc__, 'Generator')
del document_dataclass


__all__ = list(locals())
