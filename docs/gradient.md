# Gradient transformations

`tjax.gradient` provides optax's transformations as pytrees. The
[README](https://github.com/NeilGirdhar/tjax/blob/main/README.md#gradient-transformations) explains
why that matters, and this page is the reference.

## Coverage of optax

`tjax.gradient` has everything optax offers for transforming gradients: every optimizer,
transformation, and combinator, including those in `optax.contrib`. Each wrapper takes the same
arguments as its optax counterpart, so `optax.adam(1e-2)` becomes `tg.Adam(1e-2)`. The test suite
checks every wrapper's parameters against optax's signatures, and its updates against optax's under
jit.

The remaining differences are deliberate:

- `optax.chain` is `ChainedGradientTransformation`, which takes a list of transformations.
- `optax.inject_hyperparams` has no wrapper because it isn't needed: hyperparameters are already
  traced, so you can replace or differentiate them directly.
- `optax.with_extra_args_support` has no wrapper because every `update` method accepts extra keyword
  arguments, such as `value`.
- `optax.optimistic_adam` is deprecated, so `OptimisticAdam` wraps its replacement,
  `optimistic_adam_v2`.

Everything else in optax works alongside tjax, such as `optax.apply_updates`.

## Transformations beyond optax

These have no optax equivalent. `SMDGradient` implements stochastic meta-descent, which adapts a
learning rate for each weight using Hessian-vector products.

The others balance several losses, and receive each loss's gradient separately:

- `MTAdam` is Adam with each loss's gradient rescaled to match the first loss's.
- `GradientStatisticsAnnealing` weights each loss against a reference gradient.
- `IMTLG` chooses weights so that the result favours no loss.
- `PCGrad` removes the components of each gradient that conflict with the others.
- `CAGrad` improves the worst-off loss while staying near the average gradient.
- `GradVac` steers each pair of gradients toward the similarity they usually have.

All but `MTAdam` decide only how to combine gradients, so they take an optimizer of your choice,
such as `tg.PCGrad(tg.Adam(1e-3), key)`.

## Using a multi-loss transformation

A multi-loss transformation differs from a single-gradient one in two ways. `init` also takes the
loss IDs, and `update` takes a mapping from each loss ID to that loss's gradient:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import tjax.gradient as tg

params = {"w": jnp.array([1.0, -2.0])}
losses = {
    "fit": lambda p: jnp.sum((p["w"] - 3.0) ** 2),
    "penalty": lambda p: 100.0 * jnp.sum(p["w"] ** 2),
}


@jax.jit
def step(transform, params, state):
    gradients = {name: jax.grad(loss)(params) for name, loss in losses.items()}
    updates, state = transform.update(gradients, state, params)
    return optax.apply_updates(params, updates), state


transform = tg.PCGrad(tg.Adam(1e-2), jr.key(0))
state = transform.init(params, tuple(losses))
for _ in range(100):
    params, state = step(transform, params, state)
```

The order of the loss IDs matters to `MTAdam` and `GradientStatisticsAnnealing`, which use the first
loss as their reference. Pass gradients with respect to the parameters that the losses share, since
`IMTLG`, `PCGrad`, and `CAGrad` treat those parameters as a single vector.

## Base classes

To write your own transformation, subclass one of these:

- `GradientTransformation` is the counterpart of `optax.GradientTransformation`, and the base class
  of every transformation of a single gradient.
- `HvpGradientTransformation` defines an update rule that receives a Hessian-vector product.
- `DiagHessianGradientTransformation` defines one that also receives the Hessian's diagonal.
- `MultiLossGradientTransformation` defines one that receives each loss's gradient separately.
- `GradientCombiner` is a multi-loss transformation that combines the gradients and passes the
  result to an inner transformation.

## Interoperating with optax

- `to_optax` converts a tjax transformation into an optax one, and `FromOptax` does the reverse.
- `Schedule` wraps an optax schedule so that it can be passed as a learning rate.
