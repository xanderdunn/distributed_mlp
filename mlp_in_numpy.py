#!/usr/bin/env python3.11

# From https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb
# Additional reference: https://github.com/rcalix1/MachineLearningFoundations/blob/main/NeuralNets/multiLayerPerceptron.py

# System
from typing import Any, cast

# Third Party
import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

LAYERS = [
    {
        "input_dim": 2,
        "output_dim": 25,
        "activation": "relu"
    },
    {
        "input_dim": 25,
        "output_dim": 50,
        "activation": "relu"
    },
    {
        "input_dim": 50,
        "output_dim": 50,
        "activation": "relu"
    },
    {
        "input_dim": 50,
        "output_dim": 25,
        "activation": "relu"
    },
    {
        "input_dim": 25,
        "output_dim": 1,
        "activation": "sigmoid"
    },
]


def init_layers(layers: list[dict[str, int]],
                seed: int = 99) -> dict[str, np.ndarray]:
    np.random.seed(seed)
    params: dict[str, np.ndarray] = {}

    for i, layer in enumerate(layers):
        layer_id = i + 1  # TODO: Can I just use i?
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]

        # initialize weight matrix
        # weights must start out randomly, if they were all the same then
        # the gradient would be the same for all values and we would be stuck
        # Multiply by 0.1 to start with small values, for which we have
        # the largest gradients and therefore faster convergence.
        params["W" +
               str(layer_id)] = np.random.randn(output_dim, input_dim) * 0.1
        # initialize bias value
        params["b" + str(layer_id)] = np.random.randn(output_dim, 1) * 0.1

    return params


# W = weight
# b = bias
# A = activation
# Z_n = system of linear functions = W . A_(n-1) + b

# activation functions
# without the activation functions, the NN would just be a combination of
# linear functions, which is a linear function


# sigmoid activation function is used on the output layer
# because its output is in [0, 1]
def sigmoid(Z: np.ndarray):
    return 1 / (1 + np.exp(-Z))


def relu(Z: np.ndarray):
    return np.maximum(0, Z)


def sigmoid_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    sig = sigmoid(Z)
    sig_prime = sig * (1 - sig)
    # The * operator or np.multiply refers to the Hadamard product, or element-wise proudct
    # For element-wise product we need the matrices to be the same shape
    return dA * sig_prime


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def single_layer_forward(A_prev: np.ndarray,
                         W_curr: np.ndarray,
                         b_curr: np.ndarray,
                         activation="relu"):
    # When we refer to matrix product we generally mean the dot product, np.dot
    # For the dot product to be defined, we need shapes (m x n) and (n x q) produces (m x q)
    # Z = W . A_(n-1) + b
    # This is a linear function y = mx + b, except that x here is non-linearly activated
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    activation_func = relu
    if activation == "sigmoid":
        activation_func = sigmoid
    else:
        assert ("Invalid activation function {}".format(activation))

    return (activation_func(Z_curr), Z_curr)


def forward(X, params, layers: list[dict[str, Any]]):
    memory: dict[str, np.ndarray] = {}
    A_curr = X

    for i, layer in enumerate(layers):
        layer_id = i + 1
        A_prev = A_curr
        activation: str = layer["activation"]
        W_curr = params["W" + str(layer_id)]
        b_curr = params["b" + str(layer_id)]
        A_curr, Z_curr = single_layer_forward(A_prev, W_curr, b_curr,
                                              activation)

        memory["A" + str(i)] = A_prev
        memory["Z" + str(layer_id)] = Z_curr

    return A_curr, memory


# cost(prediction, label)
# Binary cross entropy
def binary_cross_entropy(Y_hat, Y) -> np.ndarray:
    num_samples = Y_hat.shape[1]
    cost = -1 / num_samples * (np.dot(Y,
                                      np.log(Y_hat).T) +
                               np.dot(1 - Y,
                                      np.log(1 - Y_hat).T))
    return np.squeeze(cost)


# convert probabilities into a class
def probs_to_classes(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.1] = 0
    return probs_


def accuracy(Y_hat, Y) -> np.floating:
    Y_hat_ = probs_to_classes(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward(dA_curr: np.ndarray,
                          W_curr: np.ndarray,
                          Z_curr: np.ndarray,
                          A_prev: np.ndarray,
                          activation="relu"):
    """
    Using the chain rule we can calcualte the values dZ, dW, db, and dA.
    """
    num_samples = A_prev.shape[1]

    activation_func = relu_backward
    if activation == "sigmoid":
        activation_func = sigmoid_backward
    else:
        assert ("Invalid activation function {}".format(activation))

    dZ_curr = activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / num_samples
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / num_samples
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return (dA_prev, dW_curr, db_curr)


def backward(Y_hat, Y, memory: dict[str, np.ndarray], params: dict[str,
                                                                   np.ndarray],
             layers: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    # init gradient descent
    # This is the gradient of the binary cross entropy activation function
    dA_prev = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_id_prev, layer in reversed(list(enumerate(layers))):
        layer_id_curr = layer_id_prev + 1
        activ_function_curr = layer["activation"]
        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_id_prev)]
        Z_curr = memory["Z" + str(layer_id_curr)]

        W_curr = params["W" + str(layer_id_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward(
            dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)
        grads["dW" + str(layer_id_curr)] = dW_curr
        grads["db" + str(layer_id_curr)] = db_curr

    return grads


def update(params: dict[str, np.ndarray], grads: dict[str, np.ndarray],
           layers: list[dict[str, Any]], learning_rate: float):
    """
    Apply the gradients to the parameters
    """

    for layer_id, _ in enumerate(layers, 1):
        params["W" +
               str(layer_id)] -= learning_rate * grads["dW" + str(layer_id)]
        params["b" +
               str(layer_id)] -= learning_rate * grads["db" + str(layer_id)]
    return params


def train(X: npt.NDArray[np.floating], Y: npt.NDArray[np.floating],
          layers: list[dict[str, Any]], epochs: int, learning_rate: float,
          random_state: int):
    params = init_layers(layers, seed=random_state)
    cost_history: list[npt.NDArray[np.floating]] = []
    accuracy_history: list[np.floating] = []

    for i in range(epochs):
        Y_hat, memory = forward(X, params, layers)

        cost = binary_cross_entropy(Y_hat, Y)
        cost_history.append(cost)
        acc = accuracy(Y_hat, Y)
        accuracy_history.append(acc)

        grads = backward(Y_hat, Y, memory, params, layers)
        params = update(params, grads, layers, learning_rate)

        # print progress on every 50th step
        if (i % 50 == 0):
            print(f"Step: {i}, cost: {cost:.5f}, accurancy: {acc:.5f}")
    return params


if __name__ == "__main__":
    # number of samples in the data set
    N_SAMPLES = 1000
    # ratio between training and test sets
    TEST_SIZE = 0.1
    NUM_STEPS = 20000
    LEARNING_RATE = 0.01
    RANDOM_SEED = 42

    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=RANDOM_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    y_train = cast(np.ndarray, y_train)
    y_train = np.transpose(y_train.reshape((y_train.shape[0], 1)))

    # train
    params = train(np.transpose(X_train),
                   y_train,
                   LAYERS,
                   NUM_STEPS,
                   LEARNING_RATE,
                   random_state=RANDOM_SEED)

    # test
    Y_test_hat, _ = forward(np.transpose(X_test), params, LAYERS)
    y_test = cast(np.ndarray, y_test)
    y_test = np.transpose(y_test.reshape((y_test.shape[0], 1)))
    acc_test = accuracy(Y_test_hat, y_test)
    print("Test set accuracy: {:.2f}".format(acc_test))
    # This shoudl produce a test accuracy of 0.98
