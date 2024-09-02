import jax.numpy as jnp
import pytest
from jax import Array, vmap

from tjax.dataclasses import DataClassModule

try:
    from flax import nnx
except ImportError:
    pytest.skip("Skipping NNX graph test", allow_module_level=True)


@pytest.mark.skip
def test_dataclass_module() -> None:
    class SomeModule(nnx.Module):
        def __init__(self, epsilon: Array) -> None:
            super().__init__()
            self.epsilon = epsilon

    class SomeDataclassModule(DataClassModule):
        def __init__(self, rngs: nnx.Rngs) -> None:
            super().__init__(rngs=rngs)
            self.sm = SomeModule(jnp.zeros(1))

    def f(m: SomeDataclassModule, x: Array) -> None:
        pass

    rngs = nnx.Rngs()
    module = SomeDataclassModule(rngs)
    z = jnp.zeros(10)
    vmap(f, in_axes=(None, 0))(module, z)
