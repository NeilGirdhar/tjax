from ._src.dataclasses.dataclass import DataclassInstance, TDataclassInstance, dataclass
from ._src.dataclasses.flax import DataClassModule, module_field
from ._src.dataclasses.helpers import as_shallow_dict, field

__all__ = ['dataclass', 'DataclassInstance', 'TDataclassInstance', 'as_shallow_dict', 'field',
           'module_field', 'DataClassModule']
