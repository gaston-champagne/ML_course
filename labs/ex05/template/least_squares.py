# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = len(y)
    M = y - tx @ w
    M = M**2


def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    l = compute_loss(y, tx, w)

    return w, l
