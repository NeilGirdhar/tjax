from typing import Hashable

from ..dataclass import dataclass, field

__all__ = ['MetaParameter']


@dataclass
class MetaParameter:

    key: Hashable = field(static=True)
