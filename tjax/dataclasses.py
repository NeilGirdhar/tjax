from ._src.dataclasses.dataclass import DataclassInstance, TDataclassInstance, dataclass
from ._src.dataclasses.helpers import as_shallow_dict, field

__all__ = ['DataclassInstance', 'TDataclassInstance', 'as_shallow_dict', 'dataclass', 'field']

try:
    import flax
except ImportError:
    pass
else:
    if flax.__version__ > '0.8':
        from ._src.dataclasses.flax import DataClassModule, module_field
        __all__ += ['DataClassModule', 'module_field']
