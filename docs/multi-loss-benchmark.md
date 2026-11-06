# Multi-loss benchmark

These results compare the multi-loss transformations on one benchmark. They show that each method
trains, but they are too limited to rank similar methods. Tune and compare methods on your own
problem before choosing one.

## Helmholtz equation

This is the two-dimensional Helmholtz benchmark from
[Wang et al., 2020](https://arxiv.org/abs/2001.04536). A physics-informed network balances a PDE
residual loss against a boundary loss.

- **Problem:** the exact solution is `sin(π x) sin(4 π y)` on the square `[-1, 1]²`, with `k = 1`.
- **Network:** a multilayer perceptron with three `tanh` hidden layers of 50 units.
- **Training:** 40,000 steps, with 128 residual points and 512 boundary points resampled each step.
- **Optimizer:** `Adam` with a learning rate of `1e-3`, decaying by 0.9 every 1,000 steps, in single
  precision. `MTAdam` uses the same schedule, and the others use it as their inner transformation.
- **Metric:** the relative `L2` error on a 101 by 101 grid, over 5 seeds. Lower is better.

Every method uses its default hyperparameters, and `residual` is the first loss.

| Method | Mean | Best seed | Worst seed |
| --- | --- | --- | --- |
| `CAGrad` | 0.015 | 0.008 | 0.025 |
| `GradientStatisticsAnnealing` | 0.018 | 0.009 | 0.039 |
| `PCGrad` | 0.019 | 0.012 | 0.032 |
| `GradVac` | 0.063 | 0.037 | 0.103 |
| `IMTLG` | 0.077 | 0.016 | 0.292 |
| No balancing (sum of the losses) | 0.373 | 0.265 | 0.409 |
| `MTAdam` | 0.493 | 0.072 | 0.716 |

For comparison, the paper reports 0.18 with no balancing, and 0.013 with its annealing method.

`MTAdam` was also unreliable with `boundary` as the first loss, with a mean of 0.586. It divides by
the largest second moment, so it may need its own learning rate. This benchmark doesn't tell us
whether a tuned `MTAdam` would do better.

To reproduce these results, run `python examples/multi_loss_benchmark.py 40000 5`. The exact
numbers vary with the random draws.

## Limitations

- This is one problem, with few seeds and untuned hyperparameters.
- The differences among the top three methods are too small to rank them.
- These runs used optax 0.2.8 and JAX 0.12 on Python 3.15.
