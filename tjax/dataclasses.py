# noqa: A005
from ._src.dataclasses.dataclass import DataclassInstance, TDataclassInstance, dataclass
from ._src.dataclasses.helpers import as_shallow_dict, field

__all__ = ['DataclassInstance', 'TDataclassInstance', 'as_shallow_dict', 'dataclass', 'field']

try:
    import flax
except ImportError:
    pass
else:
    flax_version = tuple(int(x) for x in flax.__version__.split('.'))
    if flax_version > (0, 10):
        from ._src.dataclasses.flax import DataClassModule, module_field
        __all__ += ['DataClassModule', 'module_field']
