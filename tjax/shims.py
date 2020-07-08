from typing import Any, Callable, Generic, Tuple, TypeVar, Union

from jax import custom_vjp as jax_custom_vjp
from jax import numpy as jnp
from jax.tree_util import tree_map

__all__ = ['custom_vjp']


R = TypeVar('R')


def as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return x


class custom_vjp(Generic[R]):
    """
    Shim class to work around https://github.com/google/jax/issues/2912.
    """
    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Union[int, Tuple[int, ...]] = (),
                 nondiff_argnums: Union[int, Tuple[int, ...]] = ()):
        static_argnums = as_tuple(static_argnums)
        nondiff_argnums = as_tuple(nondiff_argnums)
        if intersection := set(static_argnums) & set(nondiff_argnums):
            raise ValueError(
                f"Arguments {intersection} cannot be both static and nondifferentiable.")
        self.nondiff_argnums = nondiff_argnums
        self.vjp = jax_custom_vjp(fun, nondiff_argnums=static_argnums)

    def defvjp(self, fwd: Callable[..., Tuple[R, Any]], bwd: Callable[..., Any]) -> None:
        def new_fwd(*args: Any) -> Tuple[R, Any]:
            zeroed_args = tuple([tree_map(jnp.zeros_like, args[i])
                                 for i in self.nondiff_argnums])
            primal, internal_residuals = fwd(*args)
            return primal, (zeroed_args, internal_residuals)

        def new_bwd(residuals: Any, output_bar: R) -> Any:
            zeroed_args, internal_residuals = residuals
            input_bar = bwd(internal_residuals, output_bar)
            input_bar = list(input_bar)
            for i, index in enumerate(self.nondiff_argnums):
                input_bar[index: index] = [zeroed_args[i]]
            return tuple(input_bar)

        self.vjp.defvjp(new_fwd, new_bwd)

    def __call__(self, *args: Any) -> R:
        return self.vjp(*args)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        # https://github.com/google/jax/issues/2483
        return self.vjp.__get__(instance, owner)
