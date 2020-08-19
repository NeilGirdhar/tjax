from ..dataclass import dataclass, field

__all__ = ['MetaParameter']


@dataclass
class MetaParameter:

    name: str = field(static=True)
