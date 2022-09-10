from typing import Type

import mypy.plugins.dataclasses
from mypy.plugin import Plugin  # pylint: disable=no-name-in-module

try:
    import flax  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    has_flax = True
else:
    has_flax = False


class CustomPlugin(Plugin):
    pass


def plugin(version: str) -> Type[CustomPlugin]:
    mypy.plugins.dataclasses.dataclass_makers.add('tjax._src.dataclasses.dataclass.dataclass')
    mypy.plugins.dataclasses.field_makers.add('tjax._src.dataclasses.helpers.field')
    if has_flax:
        mypy.plugins.dataclasses.dataclass_makers.add('flax.struct.dataclasses.dataclass')
        mypy.plugins.dataclasses.field_makers.add('flax.struct.dataclasses.field')
    return CustomPlugin
