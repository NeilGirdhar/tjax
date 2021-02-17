"""
This library implements a variety of tools for the differential programming library
[JAX](https://github.com/google/jax).
"""
from .annotations import *
from .color_stub import *
from .cotangent_tools import *
from .dataclass import *
from .dataclass_patch import *
from .display import *
from .dtypes import *
from .generator import *
from .graph import *
from .leaky_integral import *
from .partial import *
from .plottable_trajectory import *
from .pytree_like import *
from .shims import *
from .testing import *
from .tools import *

__pdoc__ = {}
__pdoc__['real_dtype'] = False
__pdoc__['complex_dtype'] = False
__pdoc__['PyTreeLike'] = False
__pdoc__['Field'] = False
__pdoc__['InitVar'] = False
__pdoc__['FrozenInstanceError'] = False

document_dataclass(__pdoc__, 'Generator')
document_dataclass(__pdoc__, 'Partial')
del document_dataclass


__all__ = list(locals())
