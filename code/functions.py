#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Function module for utility and data access functions.}

{
    MIT License

    Copyright (c) 2018 maziarraissi

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
}
"""

# -------- IMPORTS --------
# Other modules
import tensorflow as tf
import numpy as np
import scipy.io

# -------- FUNCTIONS --------
def read_data(path):
    """Reads the data from a .mat file.

    Args:
        path (str): The path to the .mat file.

    Returns:
        x (np.ndarray): The x values.
        t (np.ndarray): The t values.
        exact (np.ndarray): The exact solution.
    """

    data = scipy.io.loadmat(path)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    exact = np.real(data['usol']).T
    x, t = np.meshgrid(x, t)
    return x, t, exact

def flatten_data(x, t, exact):
    """Flattens the data.

    Args:
        x (np.ndarray): The x values.
        t (np.ndarray): The t values.
        exact (np.ndarray): The exact solution.

    Returns:
        x_star (np.ndarray): The flattened x and t values.
        u_star (np.ndarray): The flattened exact solution.
    """

    x_star = np.hstack((x.flatten()[:, None], t.flatten()[:, None]))
    u_star = exact.flatten()[:, None]
    return x_star, u_star

def seed_random(seed):
    """Seeds the random number generators.

    Args:
        seed (int): The seed value.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)

def select_data(x_star, u_star, n_u, noise):
    """Selects the training data randomly.

    Args:
        x_star (np.ndarray): The flattened x and t values.
        u_star (np.ndarray): The flattened exact solution.
        n_u (int): The number of training data points.
        noise (float): The noise level.

    Returns:
        x_u_train (np.ndarray): The selected x and t values.
        u_train (np.ndarray): The selected exact solution.
    """

    idx = np.random.choice(x_star.shape[0], n_u, replace=False)
    x_u_train = x_star[idx, :]
    u_train = u_star[idx, :] + noise * np.std(u_star) * np.random.randn(n_u, 1)
    return x_u_train, u_train

def define_collocation_points(n_f, min_x, max_x, min_t, max_t):
    """Defines the collocation points.

    Args:
        n_f (int): The number of collocation points.
        min_x (float): The minimum x value.
        max_x (float): The maximum x value.
        min_t (float): The minimum t value.
        max_t (float): The maximum t value.

    Returns:
        x_f_train (np.ndarray): The collocation points.
    """

    x_f = np.linspace(min_x, max_x, int(np.sqrt(n_f)))[:, None]
    t_f = np.linspace(min_t, max_t, int(np.sqrt(n_f)))[:, None]
    x_f, t_f = np.meshgrid(x_f, t_f)
    x_f_train = np.hstack((x_f.flatten()[:, None], t_f.flatten()[:, None]))
    return x_f_train

def calculate_error(u_pred, u):
    """Calculates the relative L2 error.

    Args:
        u_pred (np.ndarray): The predicted solution.
        u (np.ndarray): The exact solution.

    Returns:
        float: The relative L2 error.
    """

    return np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
