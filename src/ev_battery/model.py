"""Multilayer perceptron trained with backpropagation, written from scratch in NumPy.

The structure follows the course modules (BAB 7 and LC 7):

* ``layer_conf`` lists the neurons per layer, for example ``(67, 32, 1)``.
* Weights are a list of matrices with shape ``(n_in + 1, n_out)``. The last row of
  every matrix is the bias, so each layer input gets an extra neuron fixed at 1.
* ``bp_fit`` trains the network, ``bp_predict`` runs the forward pass.
* Activation derivatives are written by hand (``sigd`` is ``sig * (1 - sig)``).

Extensions needed for this project:

* Regression output: the output neuron is linear by default.
* Choice of hidden activation (sigmoid, tanh, ReLU) and loss (MSE, MAE, Huber).
* Mini-batch updates. ``batch_size=1`` with ``shuffle=False`` and a sigmoid output
  reproduces the per-sample updates of the course ``bp_fit`` exactly.
* Plain gradient descent (as in the course) or Adam as a comparison.
* Optional per-sample weights, used for SOH-band balancing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Activation functions and their derivatives (with respect to the net input)
# ---------------------------------------------------------------------------


def sig(x):
    """Sigmoid, clipped to avoid overflow in ``exp`` for very negative inputs."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigd(x):
    s = sig(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanhd(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relud(x):
    return (x > 0).astype(x.dtype)


def linear(x):
    return x


def lineard(x):
    return np.ones_like(x)


ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sig, sigd),
    "tanh": (tanh, tanhd),
    "relu": (relu, relud),
    "linear": (linear, lineard),
}
LOSSES = ("mse", "mae", "huber")
OPTIMIZERS = ("gd", "adam")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _weights_or_ones(sample_weight, n):
    if sample_weight is None:
        return np.ones(n)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.shape != (n,) or (sample_weight < 0).any() or sample_weight.sum() <= 0:
        raise ValueError("sample_weight must be non-negative with one value per sample.")
    return sample_weight


def loss_value(name, output, target, sample_weight=None, huber_delta=1.0):
    """Weighted mean loss. MSE is reported as mean(e^2), the way the course prints it."""
    e = np.asarray(output, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    weight = _weights_or_ones(sample_weight, len(e))
    if name == "mse":
        per_sample = e**2
    elif name == "mae":
        per_sample = np.abs(e)
    elif name == "huber":
        small = np.abs(e) <= huber_delta
        per_sample = np.where(small, 0.5 * e**2, huber_delta * (np.abs(e) - 0.5 * huber_delta))
    else:
        raise ValueError(f"Unsupported loss {name!r}.")
    return float(np.sum(weight * per_sample) / np.sum(weight))


def loss_gradient(name, output, target, sample_weight=None, huber_delta=1.0):
    """Gradient of the batch loss with respect to every output value.

    As in the course, the MSE update uses the error ``e`` itself, which is the
    derivative of ``0.5 * e^2``. The batch gradient is the weighted mean.
    """
    e = output - target
    weight = _weights_or_ones(sample_weight, len(e))
    if name == "mse":
        g = e
    elif name == "mae":
        g = np.sign(e)
    elif name == "huber":
        g = np.clip(e, -huber_delta, huber_delta)
    else:
        raise ValueError(f"Unsupported loss {name!r}.")
    return g * (weight / np.sum(weight))[:, None]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def init_weights(layer_conf, seed=42):
    """Uniform random weights, one matrix per layer with the bias as the last row.

    The course draws from a fixed range (``uniform(-0.5, 0.5)`` in LC 7). Here the
    range shrinks with layer width, ``sqrt(6 / (n_in + n_out))``, so that sigmoid
    units with 67 inputs do not start saturated.
    """
    if len(layer_conf) < 2 or any(int(n) != n or n < 1 for n in layer_conf):
        raise ValueError("layer_conf needs at least an input and an output layer of positive sizes.")
    rng = np.random.default_rng(seed)
    weights = []
    for n_in, n_out in zip(layer_conf[:-1], layer_conf[1:]):
        limit = np.sqrt(6 / (n_in + n_out))
        w = rng.uniform(-limit, limit, (n_in + 1, n_out))
        w[-1] = 0.0  # bias row starts at zero
        weights.append(w)
    return weights


def add_bias(a):
    """Append the bias neuron (constant 1) to a batch of activations."""
    return np.hstack([a, np.ones((len(a), 1))])


def forward(X, w, activation="sigmoid", output_activation="linear"):
    """Forward pass. Returns the net inputs and activations of every layer."""
    act, _ = ACTIVATIONS[activation]
    out_act, _ = ACTIVATIONS[output_activation]
    activations = [np.asarray(X, dtype=np.float64)]
    nets = []
    last = len(w) - 1
    for layer, weight in enumerate(w):
        net = add_bias(activations[-1]) @ weight
        nets.append(net)
        activations.append(out_act(net) if layer == last else act(net))
    return nets, activations


def backward(nets, activations, w, output_gradient, activation="sigmoid", output_activation="linear"):
    """Backpropagation: gradient of the loss with respect to every weight matrix.

    ``delta`` of a layer is the loss gradient with respect to its net input. It is
    propagated to the previous layer through the weights without the bias row,
    then multiplied by the derivative of that layer's activation (chain rule).
    """
    _, actd = ACTIVATIONS[activation]
    _, out_actd = ACTIVATIONS[output_activation]
    gradients = [None] * len(w)
    delta = output_gradient * out_actd(nets[-1])
    for L in range(len(w) - 1, -1, -1):
        gradients[L] = add_bias(activations[L]).T @ delta
        if L > 0:
            delta = (delta @ w[L][:-1].T) * actd(nets[L - 1])
    return gradients


def bp_predict(X, w, activation="sigmoid", output_activation="linear", batch_size=8192):
    """Predict with trained weights. Returns a vector for a single output neuron."""
    X = np.asarray(X, dtype=np.float64)
    parts = []
    for start in range(0, len(X), batch_size):
        _, activations = forward(X[start : start + batch_size], w, activation, output_activation)
        parts.append(activations[-1])
    out = np.vstack(parts)
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite predictions.")
    return out[:, 0] if out.shape[1] == 1 else out


def n_parameters(w):
    return int(sum(weight.size for weight in w))


@dataclass
class AdamState:
    """First and second moment estimates for Adam, one pair per weight matrix."""

    m: list
    v: list
    t: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    @classmethod
    def for_weights(cls, w):
        return cls([np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w])


def update_weights(w, gradients, learn_rate, optimizer="gd", state=None):
    """Apply one update in place. ``gd`` is the course rule ``w = w - lr * gradient``."""
    if optimizer == "gd":
        for weight, grad in zip(w, gradients):
            weight -= learn_rate * grad
        return state
    if optimizer == "adam":
        state.t += 1
        for i, grad in enumerate(gradients):
            state.m[i] = state.beta1 * state.m[i] + (1 - state.beta1) * grad
            state.v[i] = state.beta2 * state.v[i] + (1 - state.beta2) * grad**2
            m_hat = state.m[i] / (1 - state.beta1**state.t)
            v_hat = state.v[i] / (1 - state.beta2**state.t)
            w[i] -= learn_rate * m_hat / (np.sqrt(v_hat) + state.eps)
        return state
    raise ValueError(f"Unsupported optimizer {optimizer!r}.")


def bp_fit(
    X,
    target,
    layer_conf,
    max_epoch=200,
    max_error=0.0,
    learn_rate=0.1,
    activation="sigmoid",
    loss="mse",
    optimizer="gd",
    batch_size=32,
    seed=42,
    shuffle=True,
    sample_weight=None,
    huber_delta=1.0,
    output_activation="linear",
    w=None,
    on_epoch_end=None,
    print_per_epoch=-1,
):
    """Train an MLP with backpropagation.

    Like the course function, training stops when ``max_epoch`` is reached or when
    the epoch MSE drops to ``max_error`` or below. ``on_epoch_end(epoch, w, mse)``
    may return ``True`` to stop early (used for validation-based early stopping).

    Returns ``(w, epoch, mse)`` exactly like the course ``bp_fit``.
    """
    if activation not in ACTIVATIONS or output_activation not in ACTIVATIONS:
        raise ValueError("Unsupported activation.")
    if loss not in LOSSES or optimizer not in OPTIMIZERS:
        raise ValueError("Unsupported loss or optimizer.")
    if learn_rate <= 0 or batch_size < 1 or max_epoch < 1 or max_error < 0:
        raise ValueError("Invalid training hyperparameters.")
    X = np.asarray(X, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(len(X), -1)
    if X.shape[1] != layer_conf[0] or target.shape[1] != layer_conf[-1]:
        raise ValueError("layer_conf does not match the input and target widths.")
    weight_all = _weights_or_ones(sample_weight, len(X))
    w = init_weights(layer_conf, seed) if w is None else [np.array(x, dtype=np.float64) for x in w]
    state = AdamState.for_weights(w) if optimizer == "adam" else None
    rng = np.random.default_rng(seed)
    epoch, mse = 0, float("inf")

    while epoch < max_epoch and mse > max_error:
        epoch += 1
        order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
        squared_error = 0.0
        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            nets, activations = forward(X[idx], w, activation, output_activation)
            output = activations[-1]
            squared_error += float(np.sum((target[idx] - output) ** 2))
            grad_out = loss_gradient(loss, output, target[idx], weight_all[idx], huber_delta)
            gradients = backward(nets, activations, w, grad_out, activation, output_activation)
            state = update_weights(w, gradients, learn_rate, optimizer, state)
        mse = squared_error / len(X)
        if not np.isfinite(mse):
            raise FloatingPointError("Training diverged (non-finite MSE). Lower the learning rate.")
        if print_per_epoch > 0 and epoch % print_per_epoch == 0:
            print(f"Epoch {epoch}, MSE: {mse}")
        if on_epoch_end is not None and on_epoch_end(epoch, w, mse):
            break
    return w, epoch, mse


@dataclass
class MLP:
    """Trained weights together with the activation they were trained with."""

    weights: list
    activation: str = "sigmoid"
    output_activation: str = "linear"

    def predict(self, X):
        return bp_predict(X, self.weights, self.activation, self.output_activation)

    @property
    def layer_conf(self):
        return tuple([self.weights[0].shape[0] - 1] + [w.shape[1] for w in self.weights])

    @property
    def n_parameters(self):
        return n_parameters(self.weights)
