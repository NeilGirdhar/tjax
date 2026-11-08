# Contributing to tjax

## Setting up

Tjax uses [uv](https://docs.astral.sh/uv/) to manage its environment and
[lefthook](https://lefthook.dev/) to run checks on each commit:

```shell
uv sync
uv run lefthook install
```

Add dependencies with `uv add` rather than by editing `pyproject.toml`.

## Checking your work

Run these before opening a pull request:

```shell
uv run lefthook run pre-commit --all-files
uv run pytest
```

The hook runs `ruff check`, `ruff format`, and `ty check` on Python files, `rumdl` on Markdown, and
`pyproject-fmt` and `taplo` on TOML.

Continuous integration runs the linters, and the tests on Python 3.13, 3.14, and 3.15. It also runs
the tests with the lowest allowed versions of the direct dependencies, so code that needs a newer
version of a dependency must raise that minimum in `pyproject.toml`.

The tests run in double precision, which `tests/conftest.py` sets up. Check numerically delicate
code in single precision too, since that is JAX's default.

## Style

- Match the surrounding code.
- Annotate all code with types.
- Keep lines to the lengths in `.editorconfig`, which is 100 for Python and Markdown.
- Write docstrings in Google style, in the order what, why, and how. Include each part only when it
  adds information.
- Prefer array API functions, such as `jnp.vecdot` to `jnp.dot`.
- Prefer abstract base classes to `Protocol`, unless structural typing is required.
- Leave no unused imports or symbols.

Write each commit message with an imperative subject of at most 72 characters. When the reason for
a change isn't obvious, add a body that explains why rather than how.

## Static and dynamic fields

In a tjax dataclass, `static=True` means static to JAX tracing. It doesn't mean non-trainable. Keep
array-bearing fields dynamic, so that JAX can trace them, and changing them doesn't recompile.

Make a field static only when a value must be known during tracing: for example, a boolean that
Python branches on, an integer that sets a shape, a dtype, or a function.

## Adding a gradient transformation

The modules in `tjax/_src/gradient` mirror optax's layout, with one file for each multi-loss method.
Export each new public class from `tjax/gradient.py`.

### Wrapping an optax transformation

Subclass `OptaxWrapper`. Its fields are the arguments of the optax function, with the same names,
order, and defaults, and its `_optax` method calls that function:

```python
@dataclass
class ScaleByExample[Weights: PyTree](OptaxWrapper[Weights]):
    """Rescale updates according to the example algorithm.

    Args:
        decay: The decay rate.
        use_bias: Whether to apply the bias correction.
    """

    decay: RealNumeric = 0.9
    use_bias: bool = field(default=True, static=True)

    @override
    def _optax(self) -> OptaxGradientTransformation:
        return optax.scale_by_example(**as_shallow_dict(self))
```

Fields that hold other transformations take tjax transformations, and `_optax` converts them with
`to_optax`.

`tests/test_optax_wrappers.py` finds such a wrapper automatically. It compares the wrapper's
parameters with optax's signature, and its updates with optax's under jit, including with each
static option changed. A wrapper that converts its fields before calling optax goes in the
`COMBINATORS` table, and the test fails until each wrapper is either compared or listed as an
exception.

If optax needs a value during tracing, the field must be static even if it looks numeric. Link a
comment on such a field to the optax issue, so that it can become dynamic once optax is fixed.

### Adding a multi-loss method

Subclass `GradientCombiner` if the method only decides how to combine gradients. Implement
`combine`, and `init_statistics` if the method keeps state between steps. Subclass
`MultiLossGradientTransformation` directly if the method needs each loss's statistics up to the
update, as `MTAdam` does.

The base class holds the loss IDs. Validate them with `check_loss_ids` in `init`, not in
`__post_init__`, since JAX reconstructs the transformation each time it unflattens it. Keep them out
of the state.

A method that uses randomness takes a required key, rather than defaulting to a fixed one.

Test a new method against an independent oracle, such as a NumPy port of its paper's algorithm, and
against a property that the paper guarantees. Check that the tests fail when you introduce a bug.

## Documentation

`README.md` is an overview, and it is also the PyPI page, so use absolute links in it. Reference
material for the gradient package belongs in `docs/gradient.md`.

## Versioning and releases

Tjax uses [EffVer](https://jacobtomlinson.dev/effver), where the version number tells users how much
effort upgrading will take. It follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for
which versions of Python and core dependencies it supports.

The changelog is written at each release, rather than with each change. Pushing a version tag
publishes the package.
