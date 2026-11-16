# Filtering

`tjax.filtering` provides online filters that summarize the history of a signal in a state of fixed
size. Each filter is a pytree of its parameters, so you can trace, differentiate, and replace them,
and its state is a separate pytree, so one filter can advance many states. The filters are exact
for signals that are rates held constant over each step, so their results don't depend on how
finely you divide time.

## The interface

Every filter has three methods:

- `init(template)` returns the initial state for signals shaped like the template, which may be any
  pytree.
- `update(state, drift, time_step=1.0)` returns the state after a step over which the signal equals
  `drift`.
- `value(state)` returns the filtered signal, with the template's structure.

```python
import jax.numpy as jnp

from tjax.filtering import ErlangChain

chain = ErlangChain.soft_window(decay=1.0, order=3)
template = {"weights": jnp.zeros((4, 2)), "bias": jnp.zeros(2)}
state = chain.init(template)  # each leaf gains a leading axis of 3 stages
for _ in range(5):
    state = chain.update(state, {"weights": jnp.ones((4, 2)), "bias": jnp.ones(2)})
trace = chain.value(state)  # shaped like the template
```

## Signals as rates

A filter treats each update's signal as a rate held over the step: `update(state, c, 2.0)` equals
two calls of `update(state, c, 1.0)`. The filters are linear systems $\dot{s} = A s + b c$, and each
step applies the exact solution of that system for a constant $c$, which is called the zero-order
hold. A signal held over a step therefore decays during that step. For example, a leaky integral
with decay $k$ adds $(1 - e^{-k})/k$ of a unit signal over a one-second step, not all of it.

The exact step is finite when a decay is zero: a leaky integral with no decay is a running sum.
Where the closed forms divide zero by zero, series replace them, and the gradients stay finite.

## Kernels

A linear filter's value is the signal weighted by a kernel $\psi(u)$, where $u$ is the time since
the signal arrived. With rates held over steps, the weight on the signal from lag $L$, the step that
ended $L$ steps before the value is read, is the kernel integrated from $L$ to $L + 1$:

$$\bar{\psi}_L = \int_L^{L+1} \psi(u) \, du.$$

Every staged filter reports these weights:

- `lag_weights(num_lags)` returns the value's weight at each lag, with shape `(num_lags,)`.
- `basis_lag_weights(num_lags)` returns each stage's weight at each lag, with shape
  `(order, num_lags)`.

A nonnegative kernel is important when the filtered signal weights evidence, as eligibility traces
do: a negative weight at some lag reverses whatever arrived at that lag. The filters below say when
their kernels are nonnegative.

## The filters

| Filter | Kernel | Memory per element | Use |
|---|---|---|---|
| `LeakyIntegral(decay)` | $e^{-k u}$ | 1 | the base case |
| `ErlangChain(decay, weights)` | $\sum_j w_j \frac{u^{j-1}}{(j-1)!} e^{-k u}$ | order | a delayed or flat window |
| `ExponentialBank(decays, weights)` | $\sum_j w_j e^{-k_j u}$ | order | long or heavy tails |
| `LinearFilter(system, input_gain, output_gain)` | any linear system | order | the general case |
| `LeakyAverage(decay)` | $e^{-k u}$, normalized | 1 | an unbiased running mean |
| `LeakyMoments(decay)` | $e^{-k u}$, normalized | $d + d^2$ per vector of length $d$ | means and second moments |
| `DelayLine(length)` | each lag separately | length | the raw history |

### Leaky integral

`LeakyIntegral` holds one value per element, with the same shape as the signal.

### Erlang chain

`ErlangChain` is a chain of leaky integrals at one decay rate $k$, where each stage integrates the
one before it. Stage $j$ alone has the kernel $u^{j-1} e^{-k u} / (j-1)!$, which peaks
$(j-1)/k$ seconds after the signal with width about $\sqrt{j-1}/k$. The weights mix the stages,
so the kernel can place its mass at a delay, which a single exponential cannot. The kernel is
nonnegative when the weights are, and with one stage and a weight of one, the chain is the leaky
integral.

`ErlangChain.soft_window(decay, order)` sets $w_j = k^{j-1}$. With $n$ stages, its kernel is the
probability that a Poisson variable with mean $k u$ is at most $n - 1$. It is near one at short lags
and falls through one half near $n/k$ seconds, so it weights the lags in that window roughly
equally.

The stages are a basis for kernels that reach about $(n - 1)/k$ seconds. To fit a kernel to data,
regress on the stages, or on the Laguerre components from `laguerre(state)`, which are orthonormal
over continuous time and so better conditioned. The Laguerre kernels take both signs, so use them to
estimate a kernel, then use nonnegative weights on the stages to apply it.

### Exponential bank

`ExponentialBank` runs leaky integrals at several decay rates in parallel. Every term peaks at no
delay, so a bank cannot form a delayed peak, but mixing slow and fast rates gives tails that decay
more slowly than any single exponential. Its kernel is nonnegative when the weights are.

### General linear filter

`LinearFilter` takes any system matrix, input gain, and output gain. Each step takes the matrix
exponential of an augmented matrix of size $n + 1$, where $n$ is the order, which gives both the
transition and the input weights (Van Loan's method), so the step stays finite when the system
matrix is singular. The specialized filters are faster, so use this for structures they don't cover.
The kernel is nonnegative when the system matrix has nonnegative off-diagonal entries and the gains
are nonnegative.

### Averages

A leaky average that starts at zero underestimates the signal until it has seen enough data.
`LeakyAverage` divides by the weight of the data seen so far, as Adam does, so its value averages
the signal from the first step. Its state is a `LeakyAverageState`.

`LeakyMoments` tracks the mean and the second moment $\mathbb{E}[x x^{\mathsf{H}}]$ of a vector
signal, which a regression needs. The second moment isn't centred; subtract the outer product of the
mean for the covariance. To track the cross moments of two signals, concatenate them.

### Delay line

`DelayLine` holds the last `length` values of the signal, newest first. Its memory grows with the
longest lag, so it suits small signals, such as one value per channel, when you want a signal's
history without assuming a kernel. Each update fills one slot, so the updates should be equally
spaced.
