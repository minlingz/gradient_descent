"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 200
    learning_rate = 1

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []

    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        loss_list.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # calculate the true probability of each token
    p_true = np.sum(encodings, axis=1) / encodings.shape[1]

    # Calculate the true log probability of observing the training data
    log_p_true = sum(
        np.log(p_true[vocabulary.index(token)] if token in vocabulary else p_true[-1])
        for token in tokens
    )

    # Calculate the minimum possible loss
    min_possible_loss = -log_p_true

    # Plot the loss values over time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="Loss")
    plt.axhline(
        y=min_possible_loss, color="r", linestyle="--", label="Minimum Possible Loss"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()

    # Calculate and plot the final token probabilities predicted by the model
    final_p_pred = normalize(torch.sigmoid(model.s)).detach().numpy()
    vocabulary[-1] = "None"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        vocabulary,
        final_p_pred.flatten(),
        label="Final Token Probabilities",
        alpha=0.8,
        width=0.4,
    )
    ax.bar(
        np.array(range(len(p_true))) + 0.4,
        p_true,
        label="True Probabilities",
        alpha=0.8,
        width=0.4,
    )
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Probabilities")
    ax.set_xticks(np.array(range(len(vocabulary))) + 0.2)
    ax.set_xticklabels(vocabulary, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()
