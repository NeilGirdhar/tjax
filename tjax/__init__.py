"""
This package provides abstract interfaces:
* tensor manipulation (TensorManipulator),
* interval arithmetic (Extrema), and
* slicing and broadcasting (Matching).

The interface is implemented using cpu and gpu routines.
"""
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
