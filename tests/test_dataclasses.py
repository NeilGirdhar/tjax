import dataclasses
import inspect

import pytest

from tjax.dataclasses import dataclass, field


@pytest.mark.parametrize(
    "field_definition",
    [
        field(),
        field(static=True),
        field(default=1),
        field(default=1, static=True),
        field(default_factory=list),
        field(default_factory=list, static=True),
    ],
)
def test_field_defaults_to_positional(field_definition: object) -> None:
    @dataclass
    class Example:
        value: object = field_definition

    dataclass_field = dataclasses.fields(Example)[0]
    parameter = inspect.signature(Example).parameters["value"]

    assert dataclass_field.kw_only is False
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_field_respects_explicit_kw_only() -> None:
    @dataclass
    class Example:
        positional: int = field(default=1, kw_only=False)
        keyword_only: int = field(default=2, kw_only=True)

    signature = inspect.signature(Example)

    assert signature.parameters["positional"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["keyword_only"].kind is inspect.Parameter.KEYWORD_ONLY


def test_field_without_default_is_required() -> None:
    @dataclass
    class Example:
        value: int = field()

    parameter = inspect.signature(Example).parameters["value"]

    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="required positional argument: 'value'"):
        Example()  # type: ignore[call-arg] # type: ignore
