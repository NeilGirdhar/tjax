"""Tests that the optax wrappers match their optax counterparts.

A simple wrapper is an ``OptaxWrapper`` whose ``_optax`` method passes its fields as keyword
arguments to one optax function.  Each such wrapper must have the same parameters as that function,
and must produce the same updates under jit.  Each combinator must have the same parameters as its
optax combinator, and ``test_combinators.py`` compares their updates.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax import tree

import tjax.gradient as tg
from tjax import JaxArray, PyTree
from tjax.dataclasses import as_shallow_dict
from tjax.gradient import OptaxWrapper

STEPS = 4
# Parameters that optax deprecates, which the wrappers deliberately omit.
DEPRECATED = {
    ("add_noise", "seed"),
    ("noisy_sgd", "seed"),
    ("differentially_private_aggregate", "seed"),
}
# Parameters that the wrappers require although optax gives them a default.  Omitting them would
# silently use key 0, so that every run gets the same randomness.
MADE_REQUIRED = {
    ("add_noise", "key"),
    ("noisy_sgd", "key"),
    ("differentially_private_aggregate", "key"),
    ("hutchinson_estimator_diag_hessian", "random_seed"),
    ("sophia", "hessian_diagonal_fn"),
}
# Parameters that the wrappers make keyword-only, as https://github.com/google-deepmind/optax/issues/1858
# proposes.
MADE_KEYWORD_ONLY = {("noisy_sgd", "eta"), ("noisy_sgd", "gamma")}


type Wrapper = type[OptaxWrapper[Any]]


def _optax_path(cls: Wrapper) -> str | None:
    match = re.search(r"return (optax(?:\.contrib)?\.\w+)\(\*\*", inspect.getsource(cls._optax))
    return None if match is None else match[1]


def _resolve(path: str) -> Callable[..., Any]:
    obj: Any = optax
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


WRAPPERS = sorted(
    (
        (cls, path)
        for name in tg.__all__
        if isinstance(cls := getattr(tg, name), type)
        and issubclass(cls, OptaxWrapper)
        and (path := _optax_path(cls)) is not None
    ),
    key=lambda pair: pair[0].__name__,
)
IDS = [cls.__name__ for cls, _ in WRAPPERS]
# Wrappers that convert their fields before calling optax, so they are not found automatically.
COMBINATORS: list[tuple[Wrapper, str]] = [
    (tg.ApplyIfFinite, "optax.apply_if_finite"),
    (tg.ConditionallyMask, "optax.conditionally_mask"),
    (tg.ConditionallyTransform, "optax.conditionally_transform"),
    (tg.Flatten, "optax.flatten"),
    (tg.Freeze, "optax.freeze"),
    (tg.Lookahead, "optax.lookahead"),
    (tg.Masked, "optax.masked"),
    (tg.MeasureWithEMA, "optax.measure_with_ema"),
    (tg.MultiSteps, "optax.MultiSteps"),
    (tg.Partition, "optax.partition"),
    (tg.SelectiveTransform, "optax.selective_transform"),
    (tg.Snapshot, "optax.snapshot"),
    (tg.Stateless, "optax.stateless"),
    (tg.StatelessWithTreeMap, "optax.stateless_with_tree_map"),
]
# Wrappers whose fields deliberately differ from the parameters of what they wrap.  FromOptax holds
# any transformation, Monitor splits optax's measures by kind, and NamedChain replaces *args.
NOT_COMPARED: set[Wrapper] = {tg.FromOptax, tg.Monitor, tg.NamedChain}


def test_every_wrapper_is_compared() -> None:
    exported = {
        cls
        for name in tg.__all__
        if isinstance(cls := getattr(tg, name), type)
        and issubclass(cls, OptaxWrapper)
        and cls is not OptaxWrapper
    }
    compared = {cls for cls, _ in WRAPPERS} | {cls for cls, _ in COMBINATORS}
    assert exported == compared | NOT_COMPARED


# Signatures ---------------------------------------------------------------------------------------
def _same_default(ours: object, theirs: object) -> bool:
    if callable(ours) or isinstance(ours, optax.GradientTransformation):
        return type(ours) is type(theirs) or (callable(ours) and callable(theirs))
    return repr(ours) == repr(theirs)


@pytest.mark.parametrize(
    ("cls", "path"), WRAPPERS + COMBINATORS, ids=[cls.__name__ for cls, _ in WRAPPERS + COMBINATORS]
)
def test_signature_matches_optax(cls: Wrapper, path: str) -> None:
    function_name = path.rsplit(".", 1)[1]
    theirs = {
        name: p
        for name, p in inspect.signature(_resolve(path)).parameters.items()
        if (function_name, name) not in DEPRECATED
    }
    ours = {f.name: f for f in dataclasses.fields(cls) if f.name != "_"}
    made_required = {name for function, name in MADE_REQUIRED if function == function_name}
    assert set(ours) == set(theirs)
    assert [n for n in ours if n not in made_required] == [
        n for n in theirs if n not in made_required
    ]
    for name, p in theirs.items():
        f = ours[name]
        keyword_only = p.kind is p.KEYWORD_ONLY or (function_name, name) in MADE_KEYWORD_ONLY
        assert f.kw_only == keyword_only, name
        if p.default is inspect.Parameter.empty or name in made_required:
            assert f.default is dataclasses.MISSING, name
            assert f.default_factory is dataclasses.MISSING, name
        else:
            default = f.default_factory() if f.default is dataclasses.MISSING else f.default  # ty: ignore[call-non-callable]
            if isinstance(default, OptaxWrapper):
                default = default._optax()  # ruff: ignore[private-member-access]
            assert _same_default(default, p.default), (name, default, p.default)


# Numerical comparison -----------------------------------------------------------------------------
REQUIRED: dict[str, object] = {
    "learning_rate": 1e-2,
    "max_delta": 0.5,
    "max_norm": 1.0,
    "threshold": 1.0,
    "clipping": 0.1,
    "max_backtracking_steps": 5,
    "max_linesearch_steps": 5,
    "l2_norm_clip": 1.0,
    "noise_multiplier": 0.1,
    "seed": 0,
    "eta": 0.01,
    "gamma": 0.55,
    "decay": 0.9,
    "step_size": 0.5,
    "step_size_fn": tg.Schedule(lambda count: 0.1 / (1.0 + count)),
    "base_optimizer": tg.SGD(0.1),
    "optimizer": tg.SGD(0.1),
    "adv_optimizer": tg.SGD(0.1),
    "inner": tg.Identity(),
    "key": jr.key(1),
    "random_seed": jr.key(2),
    "hessian_diagonal_fn": tg.HutchinsonEstimatorDiagHessian(jr.key(2)),
}
# Values for optional numeric arguments whose default is None.
OPTIONAL: dict[str, object] = {
    "momentum": 0.9,
    "weight_decay": 1e-3,
    "weight_decay_rate": 1e-3,
    "beta3": 0.99,
    "init_estim_sq_dist": 1e-6,
    "consistent_rms": 0.2,
    "adam_learning_rate": 1e-3,
    "max_learning_rate": 1.0,
    "approx_dec_rtol": 1e-6,
    "clipping_threshold": 1.0,
    "clip_threshold": 1.0,
}
# Wrappers whose optional arguments do not accept the values above.
NO_OPTIONAL = {"Rprop", "ScaleByRprop"}
# The extra update arguments that each wrapper needs; the rest receive only the loss value.
LINESEARCH_ARGS = ("value", "grad", "value_fn")
EXTRA_ARGS: dict[str, tuple[str, ...]] = {
    "LBFGS": LINESEARCH_ARGS,
    "ScaleByBacktrackingLinesearch": LINESEARCH_ARGS,
    "ScaleByZoomLinesearch": LINESEARCH_ARGS,
    "Sophia": ("obj_fn",),
    "HutchinsonEstimatorDiagHessian": ("obj_fn",),
    "SAM": (),
}
# Wrappers that need gradients with a leading batch axis.
PER_EXAMPLE = {"DPSGD", "DifferentiallyPrivateAggregate"}
# Wrappers whose default dimension numbers require the parameters to be a single matrix.
MATRICES_ONLY = {"ScaleByMuon"}


def _loss(params: PyTree) -> JaxArray:
    return sum((jnp.sum(jnp.square(x - 0.5)) for x in tree.leaves(params)), jnp.asarray(0.0))


def _params(name: str) -> PyTree:
    key_w, key_b = jr.split(jr.key(0))
    params = {"w": jr.normal(key_w, (4, 3)), "b": jr.normal(key_b, (3,))}
    return params["w"] if name in MATRICES_ONLY else params


def _noise(name: str) -> list[dict[str, JaxArray]]:
    keys = jr.split(jr.key(3), STEPS)
    return [tree.map(lambda x, k=k: 0.1 * jr.normal(k, x.shape), _params(name)) for k in keys]


def _gradient(name: str, grad: PyTree, noise: PyTree) -> PyTree:
    gradient = tree.map(jnp.add, grad, noise)
    if name in PER_EXAMPLE:
        return tree.map(lambda x: jnp.stack([x, -0.5 * x]), gradient)
    return gradient


def _construct(cls: Wrapper, *, optional: bool) -> OptaxWrapper[Any]:
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name == "_":
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        if required:
            kwargs[f.name] = REQUIRED[f.name]
        elif optional and f.default is None and f.name in OPTIONAL:
            kwargs[f.name] = OPTIONAL[f.name]
    return cls(**kwargs)


def _to_optax(tx: OptaxWrapper[Any], path: str) -> optax.GradientTransformation:
    kwargs = {
        name: value._optax() if isinstance(value, OptaxWrapper) else value  # ruff: ignore[private-member-access]
        for name, value in as_shallow_dict(tx).items()
    }
    return _resolve(path)(**kwargs)


def _extra_args(name: str, params: PyTree) -> tuple[PyTree, dict[str, object]]:
    value, grad = jax.value_and_grad(_loss)(params)
    available = {"value": value, "grad": grad, "value_fn": _loss, "obj_fn": _loss}
    return grad, {arg: available[arg] for arg in EXTRA_ARGS.get(name, ("value",))}


def _run_tjax(tx: OptaxWrapper[Any]) -> list[PyTree]:
    name = type(tx).__name__

    @jax.jit
    def step(
        tx: OptaxWrapper[Any], noise: PyTree, state: PyTree, params: PyTree
    ) -> tuple[PyTree, PyTree]:
        grad, extra_args = _extra_args(name, params)
        return tx.update(_gradient(name, grad, noise), state, params, **extra_args)

    params = _params(name)
    state = tx.init(params)
    all_updates = []
    for noise in _noise(name):
        updates, state = step(tx, noise, state, params)
        all_updates.append(updates)
    return all_updates


def _run_optax(name: str, tx: optax.GradientTransformation) -> list[PyTree]:
    params = _params(name)
    state = tx.init(params)
    all_updates = []
    for noise in _noise(name):
        grad, extra_args = _extra_args(name, params)
        gradient = _gradient(name, grad, noise)
        if isinstance(tx, optax.GradientTransformationExtraArgs):
            updates, state = tx.update(gradient, state, params, **extra_args)
        else:
            updates, state = tx.update(gradient, state, params)
        all_updates.append(updates)
    return all_updates


@pytest.mark.parametrize("optional", [False, True], ids=["defaults", "optional-set"])
@pytest.mark.parametrize(("cls", "path"), WRAPPERS, ids=IDS)
def test_matches_optax_under_jit(cls: Wrapper, path: str, *, optional: bool) -> None:
    if optional and cls.__name__ in NO_OPTIONAL:
        pytest.skip("no optional arguments to set")
    tx = _construct(cls, optional=optional)
    chex.assert_trees_all_close(
        _run_tjax(tx), _run_optax(cls.__name__, _to_optax(tx, path)), rtol=1e-5, atol=1e-6
    )


# Static options -----------------------------------------------------------------------------------
# These are chosen by name and default rather than by the field's static marker, so that a field
# that should be static but isn't still gets compared, and fails under jit.
def _matrices_only(params: PyTree) -> PyTree:
    return tree.map(lambda x: x.ndim > 1, params)


# Non-default values for static fields that are not booleans or dtypes.
STATIC_OPTIONS: dict[str, list[object]] = {
    "axis": [0],
    "init_step": [("distance", 1e-3), ("learning_rate", 1e-3)],
    "initial_guess_strategy": ["one"],
    "mask": [_matrices_only],
    "mode": ["smooth", "refined"],
    "preconditioning": ["spectral", "aol", "schatten"],
    "trust_ratio_mask": [_matrices_only],
    "variant": ["sps+"],
    "warmup_steps": [2],
    "weight_decay_mask": [_matrices_only],
}
# SAM's opaque mode needs a gradient function that this comparison does not supply.
NO_STATIC_OPTIONS = {("SAM", "opaque_mode")}


def _static_options(f: dataclasses.Field[Any]) -> list[object]:
    if isinstance(f.default, bool):
        return [not f.default]
    if f.name.endswith("dtype"):
        return [jnp.float32]
    if f.name == "ns_coeffs" and "str" in str(f.type):
        return ["standard"]
    return STATIC_OPTIONS.get(f.name, [])


STATIC_CASES = [
    pytest.param(cls, path, f.name, value, id=f"{cls.__name__}-{f.name}-{i}")
    for cls, path in WRAPPERS
    for f in dataclasses.fields(cls)
    if f.name != "_" and (cls.__name__, f.name) not in NO_STATIC_OPTIONS
    for i, value in enumerate(_static_options(f))
]


def test_static_options_are_found() -> None:
    assert len(STATIC_CASES) > 100  # ruff: ignore[magic-value-comparison]


@pytest.mark.parametrize(("cls", "path", "name", "value"), STATIC_CASES)
def test_static_option_matches_optax(cls: Wrapper, path: str, name: str, value: object) -> None:
    tx = dataclasses.replace(_construct(cls, optional=False), **{name: value})
    chex.assert_trees_all_close(
        _run_tjax(tx), _run_optax(cls.__name__, _to_optax(tx, path)), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (tg.AddNoise, (0.01, 0.55)),
        (tg.NoisySGD, (1e-2,)),
        (tg.DifferentiallyPrivateAggregate, (1.0, 0.1)),
        (tg.HutchinsonEstimatorDiagHessian, ()),
        (tg.Sophia, (1e-2,)),
    ],
)
def test_randomness_must_be_explicit(cls: Callable[..., object], args: tuple[float, ...]) -> None:
    with pytest.raises(TypeError, match="missing"):
        cls(*args)
