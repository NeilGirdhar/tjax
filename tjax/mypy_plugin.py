from typing import Type

import mypy.plugins.dataclasses
from mypy.plugin import Plugin  # pylint: disable=no-name-in-module


class CustomPlugin(Plugin):
    pass


def plugin(version: str) -> Type[CustomPlugin]:
    # pylint: disable=c-extension-no-member
    mypy.plugins.dataclasses.dataclass_makers.add('tjax._src.dataclasses.dataclass.dataclass')
    # mypy.plugins.dataclasses.field_makers.add('tjax._src.dataclasses.helpers.field')
    return CustomPlugin
