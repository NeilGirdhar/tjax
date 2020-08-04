from typing import Any, Callable, Generic, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.tree_util import Partial, tree_map

__all__ = ['jit', 'custom_jvp', 'custom_vjp']


R = TypeVar('R')
F = TypeVar('F', bound=Callable[..., Any])


def as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return x


def jit(func: F, **kwargs: Any) -> F:
    retval = jax.jit(func, **kwargs)
    if hasattr(func, "__isabstractmethod__"):
        retval.__isabstractmethod__ = func.__isabstractmethod__  # type: ignore
    return retval


class custom_vjp(Generic[R]):
    """
    Set up a JAX-transformable function for a custom VJP rule definition.

    This class is meant to be used as a function decorator. Instances are callables that behave
    similarly to the underlying function to which the decorator was applied, except when a
    reverse-mode differentiation transformation (like `jax.grad()`) is applied, in which case a
    custom user-supplied VJP rule function is used instead of tracing into and performing automatic
    differentiation of the underlying function’s implementation. There is a single instance method,
    defvjp, which defines the custom VJP rule.

    This decorator precludes the use of forward-mode automatic differentiation.

    This is a shim class to work around an
    [issue with JAX's custom_vjp](https://github.com/google/jax/issues/2912).  It provides both:

    - static arguments, and
    - nondifferentiable arguments.

    Static arguments are passed in to both the forward and the backward pass.  They must be
    hashable.  Different values for static arguments will generate recompilation.

    The generated backward pass will generate zeroed-out cotangents.  Ideally, no corresponding
    cotangents would be created, but such a change would have to be done in JAX itself.

    For example::

        from tjax import custom_vjp
        from jax import numpy as jnp

        @partial(custom_vjp, nondiff_argnums=2)
        def f(x, y, z):
        return jnp.sin(x) * y + z

        def f_fwd(x, y, z):
        return f(x, y, z), (jnp.cos(x), jnp.sin(x), y)

        def f_bwd(residuals, output_bar):
        cos_x, sin_x, y = residuals
        x_bar = cos_x * output_bar * y
        y_bar = sin_x * output_bar
        # z_bar is not returned because it's nondifferentiable.
        return x_bar, y_bar

        f.defvjp(f_fwd, f_bwd)
    """
    def __init__(self,
                 fun: Callable[..., R],
                 static_argnums: Union[int, Tuple[int, ...]] = (),
                 nondiff_argnums: Union[int, Tuple[int, ...]] = ()):
        """
        Args:
            fun: the function to decorate.
            static_argnums: The indices of the static arguments.
            nondiff_argnums: The indices of the nondifferentiable arguments.
        """
        static_argnums = as_tuple(static_argnums)
        nondiff_argnums = as_tuple(nondiff_argnums)
        intersection = set(static_argnums) & set(nondiff_argnums)
        if intersection:
            raise ValueError(
                f"Arguments {intersection} cannot be both static and nondifferentiable.")
        self.nondiff_argnums = nondiff_argnums
        self.vjp = jax.custom_vjp(fun, nondiff_argnums=static_argnums)

    def defvjp(self, fwd: Callable[..., Tuple[R, Any]], bwd: Callable[..., Any]) -> None:
        """
        Implement the custom forward and backward passes of the custom derivative.

        Args:
            fwd: The custom forward pass.
            bwd: The custom backward pass.  Cotangents for the nondifferentiable arguments should
                not be provided by the user-provided backward pass.
        """
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
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)


class custom_jvp(jax.custom_jvp, Generic[R]):
    """Set up a JAX-transformable function for a custom JVP rule definition.

    This class is meant to be used as a function decorator. Instances are
    callables that behave similarly to the underlying function to which the
    decorator was applied, except when a differentiation transformation (like
    :py:func:`jax.jvp` or :py:func:`jax.grad`) is applied, in which case a custom user-supplied
    JVP rule function is used instead of tracing into and performing automatic
    differentiation of the underlying function's implementation. There is a single
    instance method, ``defjvp``, which defines the custom JVP rule.

    For example::

    import jax.numpy as jnp

    @jax.custom_jvp
    def f(x, y):
        return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
        x, y = primals
        x_dot, y_dot = tangents
        primal_out = f(x, y)
        tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
        return primal_out, tangent_out

    For a more detailed introduction, see the tutorial_.
    """

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., R]:
        if instance is None:
            return self
        # Create a partial function application corresponding to a bound method.
        return Partial(self, instance)
