from __future__ import annotations

import jax.numpy as jnp
from jax import grad

from tjax._src.abstract_method_decorators import (  # noqa: PLC2701
    JaxAbstractClass,
    abstract_custom_jvp,
    abstract_jit,
)
from tjax._src.dataclasses.dataclass import dataclass  # noqa: PLC2701
from tjax._src.function_markers import (  # noqa: PLC2701
    abstract_custom_jvp_marker,
    abstract_jit_marker,
    jit_marker,
)
from tjax._src.shims import jit  # noqa: PLC2701

EXPECTED_JIT_VALUE = 3.0
EXPECTED_CUSTOM_GRADIENT = 4.0
EXPECTED_SUPER_VALUE = 4.0
EXPECTED_INHERITED_VALUE = 2.0


def test_abstract_jit_marker_survives_intermediate_override() -> None:
    @dataclass
    class Base(JaxAbstractClass):
        @abstract_jit
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            raise NotImplementedError

    @dataclass
    class Intermediate(Base):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

    @dataclass
    class Concrete(Intermediate):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 2

    assert getattr(Intermediate.__dict__["value"], abstract_jit_marker) == {}
    assert getattr(Concrete.__dict__["value"], abstract_jit_marker) == {}
    assert Concrete().value(jnp.asarray(1.0)) == EXPECTED_JIT_VALUE


def test_abstract_jit_intermediate_override_supports_super() -> None:
    @dataclass
    class Base(JaxAbstractClass):
        @abstract_jit
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            raise NotImplementedError

    @dataclass
    class Intermediate(Base):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

    @dataclass
    class Concrete(Intermediate):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return super().value(x) + 2

    assert getattr(Concrete.__dict__["value"], abstract_jit_marker) == {}
    assert Concrete().value(jnp.asarray(1.0)) == EXPECTED_SUPER_VALUE


def test_concrete_marked_method_is_rewrapped_when_inherited() -> None:
    @dataclass
    class Base(JaxAbstractClass):
        @abstract_jit
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            raise NotImplementedError

    @dataclass
    class Intermediate(Base):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

    @dataclass
    class Concrete(Intermediate):
        pass

    assert "value" in Concrete.__dict__
    assert Concrete.__dict__["value"] is not Intermediate.__dict__["value"]
    assert getattr(Concrete.__dict__["value"], abstract_jit_marker) == {}
    assert Concrete().value(jnp.asarray(1.0)) == EXPECTED_INHERITED_VALUE


def test_concrete_jitted_method_is_rewrapped_when_inherited() -> None:
    @dataclass
    class Base(JaxAbstractClass):
        @jit
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

    @dataclass
    class Concrete(Base):
        pass

    assert "value" in Concrete.__dict__
    assert Concrete.__dict__["value"] is not Base.__dict__["value"]
    assert getattr(Concrete.__dict__["value"], jit_marker) == {}
    assert Concrete().value(jnp.asarray(1.0)) == EXPECTED_INHERITED_VALUE


def test_jitted_override_keeps_jit_when_parent_method_is_jitted() -> None:
    @dataclass
    class Base(JaxAbstractClass):
        @jit
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

    @dataclass
    class Concrete(Base):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x + 2

    assert getattr(Concrete.__dict__["value"], jit_marker) == {}
    assert Concrete().value(jnp.asarray(1.0)) == EXPECTED_JIT_VALUE


def test_abstract_custom_jvp_marker_survives_intermediate_override() -> None:
    def value_jvp(
        primals: tuple[CustomJvpBase, jnp.ndarray],
        tangents: tuple[CustomJvpBase, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self, x = primals
        _, x_dot = tangents
        return self.value(x), 4.0 * x_dot

    @dataclass
    class CustomJvpBase(JaxAbstractClass):
        @abstract_custom_jvp(value_jvp)
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            raise NotImplementedError

    @dataclass
    class Intermediate(CustomJvpBase):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x**2

    @dataclass
    class Concrete(Intermediate):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x**3

    marker = getattr(Concrete.__dict__["value"], abstract_custom_jvp_marker)
    assert marker[0] is value_jvp
    assert marker[1] == ()
    assert grad(Concrete().value)(jnp.asarray(2.0)) == EXPECTED_CUSTOM_GRADIENT


def test_concrete_custom_jvp_method_is_rewrapped_when_inherited() -> None:
    def value_jvp(
        primals: tuple[CustomJvpBase, jnp.ndarray],
        tangents: tuple[CustomJvpBase, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self, x = primals
        _, x_dot = tangents
        return self.value(x), 4.0 * x_dot

    @dataclass
    class CustomJvpBase(JaxAbstractClass):
        @abstract_custom_jvp(value_jvp)
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            raise NotImplementedError

    @dataclass
    class Intermediate(CustomJvpBase):
        def value(self, x: jnp.ndarray) -> jnp.ndarray:
            return x**2

    @dataclass
    class Concrete(Intermediate):
        pass

    assert "value" in Concrete.__dict__
    assert Concrete.__dict__["value"] is not Intermediate.__dict__["value"]
    marker = getattr(Concrete.__dict__["value"], abstract_custom_jvp_marker)
    assert marker[0] is value_jvp
    assert marker[1] == ()
    assert grad(Concrete().value)(jnp.asarray(2.0)) == EXPECTED_CUSTOM_GRADIENT
