from typing import Type

import mypy.plugins.dataclasses
from mypy.plugin import Plugin


class CustomPlugin(Plugin):
    pass


def plugin(version: str) -> Type[CustomPlugin]:
    mypy.plugins.dataclasses.dataclass_makers.add('tjax._src.dataclasses.dataclass.dataclass')
    # mypy.plugins.dataclasses.field_makers.add('tjax._src.dataclasses.helpers.field')
    return CustomPlugin
