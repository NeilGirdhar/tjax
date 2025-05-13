"""Meta-learning.

Using tjax.gradient is much simpler than optax since we don't need inject_hyperparams.

Compare this example with
https://optax.readthedocs.io/en/latest/_collections/examples/meta_learning.html
"""
import operator
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jss
import matplotlib.pyplot as plt
from jax import tree

from tjax import JaxArray, RngStream
from tjax.dataclasses import dataclass
from tjax.gradient import Adam, GenericGradientState, GradientState, GradientTransformation, RMSProp

T = TypeVar('T')


@dataclass
class InferenceState:
    theta: JaxArray  # Model parameters.
    state: GradientState  # Gradient state of the model parameters.


@dataclass
class OuterInferenceState:
    inference_state: InferenceState
    eta: JaxArray  # Hyperparameters.
    meta_state: GenericGradientState  # Gradient state of the hyperparameters.


@dataclass
class TrainingExample:
    observation: JaxArray
    target: JaxArray


def apply_updates(parameters: T, updates: T) -> T:
    return tree.map(operator.add, parameters, updates)


def generate_example(stream: RngStream) -> TrainingExample:
    observation = jr.uniform(stream.key(), minval=0.0, maxval=10.0)
    target = 10.0 * observation + jr.normal(stream.key())
    return TrainingExample(observation, target)


def model(theta: JaxArray, x: JaxArray) -> JaxArray:
    # This is the simplest possible model.
    return x * theta


def loss(theta: JaxArray, training_example: TrainingExample) -> JaxArray:
    return jnp.sum(jnp.square(training_example.target - model(theta, training_example.observation)))


def step(opt: GradientTransformation[Any, JaxArray],
         inference_state: InferenceState,
         training_example: TrainingExample,
         ) -> InferenceState:
    gradient = jax.grad(loss)(inference_state.theta, training_example)
    updates, state = opt.update(gradient, inference_state.state, inference_state.theta)
    theta = apply_updates(inference_state.theta, updates)
    return InferenceState(theta, state)


@jax.jit
def outer_loss(eta: JaxArray,
               inference_state: InferenceState,
               training_examples: list[TrainingExample],
               ) -> tuple[JaxArray, InferenceState]:
    # Eta is the logit of the learning rate.
    opt = RMSProp(learning_rate=jss.expit(eta))
    # Use all but one sample to train theta.
    for training_example in training_examples[:-1]:
        inference_state = step(opt, inference_state, training_example)
    # Use the final sample to train eta.
    training_example = training_examples[-1]
    return loss(inference_state.theta, training_example), inference_state


@jax.jit
def outer_step(outer_inference_state: OuterInferenceState,
               training_examples: list[tuple[JaxArray, JaxArray]],
               meta_opt: GradientTransformation[Any, JaxArray],
               ) -> OuterInferenceState:
    g_outer_loss = jax.grad(outer_loss, has_aux=True)
    gradient, inference_state = g_outer_loss(outer_inference_state.eta,
                                             outer_inference_state.inference_state,
                                             training_examples)

    meta_updates, meta_state = meta_opt.update(gradient, outer_inference_state.meta_state, None)
    new_eta = apply_updates(outer_inference_state.eta, meta_updates)
    return OuterInferenceState(inference_state, new_eta, meta_state)


def create_state() -> tuple[OuterInferenceState, GradientTransformation[Any, JaxArray]]:
    init_learning_rate = jnp.array(0.1)
    meta_learning_rate = jnp.array(0.03)
    opt = RMSProp(learning_rate=init_learning_rate)
    meta_opt = Adam(learning_rate=meta_learning_rate)
    # Eta is the logit of the learning rate.
    eta = jss.logit(init_learning_rate)
    theta = jax.random.normal(jax.random.PRNGKey(42))
    inference_state = InferenceState(theta=theta, state=opt.init(theta))
    outer_inference_state = OuterInferenceState(inference_state=inference_state, eta=eta,
                                                meta_state=meta_opt.init(eta))
    return outer_inference_state, meta_opt


def run() -> None:
    outer_inference_state, meta_opt = create_state()
    stream = RngStream(jax.random.key(0))
    learning_rates = []
    thetas = []

    for _ in range(2000):
        training_examples = [generate_example(stream) for _ in range(7)]
        outer_inference_state = outer_step(outer_inference_state, training_examples, meta_opt)
        learning_rates.append(jss.expit(outer_inference_state.eta))
        thetas.append(outer_inference_state.inference_state.theta)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Meta-learning RMSProp\'s learning rate')
    plt.xlabel('Step')

    ax1.semilogy(range(len(learning_rates)), learning_rates)
    ax1.set(ylabel='Learning rate')
    ax1.label_outer()

    plt.xlabel('Number of updates')
    ax2.semilogy(range(len(thetas)), thetas)

    ax2.label_outer()
    ax2.set(ylabel='Theta')

    plt.show()


run()
