#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Function module for plotting functions.}

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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------- FUNCTIONS --------
def _imshow_subplot(ax, data, title, x_label, y_label, label, x_ticks, y_ticks,
                    x_tick_label, y_tick_label, cmap='viridis', vmin=-1, vmax=1):
    # Plot the data
    sns.heatmap(data, ax=ax, cbar=True, cmap=cmap, label=label,
                cbar_kws={'label': 'Magnitude'}, vmin=vmin, vmax=vmax)

    # Set the plot labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_tick_label)
    ax.set_yticklabels(y_tick_label)

def _lineplot_subplot(ax, x_data, title, x_label, y_label, label,
                      x_ticks=None, y_ticks=None, x_tick_label=None,
                      y_tick_label=None, y_scale='linear'):
    # Plot the data
    sns.lineplot(data=x_data, ax=ax, color="tab:orange", label=label)

    # Set the plot labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_label)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_label)

    # Set the y-scale
    ax.set_yscale(y_scale)

def plot_solution(u_pred, exact, path, style='whitegrid', figsize=(15, 5), dpi=300):
    """Plotting function that compares the solution of the PINN with the exact one.

    Args:
        u_pred (np.ndarray): Predicted velocities.
        exact (np.ndarray): Exact velocities.
        path (str): Save path.
        style (str, optional): Plotting style. Defaults to 'whitegrid'.
        figsize (tuple, optional): Figure dimensions. Defaults to (15, 5).
        dpi (int, optional): Output DPI. Defaults to 300.
    """

    # Calculate the plotting data
    pred = u_pred.reshape(exact.shape).T
    exact = exact.T
    error = np.abs(np.abs(exact - pred))

    # Configure the plot style
    sns.set_theme(style=style)

    # Setup for 2x2 grid plots
    _, axs = plt.subplots(2, 2, figsize=figsize)

    # Plot the predicted solution
    _imshow_subplot(axs[0, 0], pred, 'Burgers Equation - Prediction',
                    'Time [s]', 'x', 'Prediction',
                    [0, 19, 39, 59, 79, 99], [0, 63, 127, 191, 255],
                    [0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.5, 0, -0.5, -1])

    # Plot the exact solution
    _imshow_subplot(axs[0, 1], exact, 'Burgers Equation - Exact',
                    'Time [s]', 'x', 'Exact',
                    [0, 19, 39, 59, 79, 99], [0, 63, 127, 191, 255],
                    [0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.5, 0, -0.5, -1])

    # Plot the error
    _imshow_subplot(axs[1, 0], error, 'Burgers Equation - Error',
                    'Time [s]', 'x', 'Error',
                    [0, 19, 39, 59, 79, 99], [0, 63, 127, 191, 255],
                    [0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.5, 0, -0.5, -1])

    # Plot the mean error
    _lineplot_subplot(axs[1, 1], error.mean(axis=0), 'Mean Error',
                      'Time [s]', 'Error', 'Mean Error',
                      x_ticks=[0, 19, 39, 59, 79, 99], x_tick_label=[0, 0.2, 0.4, 0.6, 0.8, 1])

    # Finalize layout
    sns.despine()
    plt.tight_layout()
    plt.legend()

    # Export the plot
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_losses(model, path, style='whitegrid', figsize=(15, 5), dpi=300):
    """Plotting function that visualizes the data, physics, and total losses.

    Args:
        model (PhysicsInformedNN): The trained model.
        path (str): Save path.
        style (str, optional): Plotting style. Defaults to 'whitegrid'.
        figsize (tuple, optional): Figure dimensions. Defaults to (15, 5).
        dpi (int, optional): Output DPI. Defaults to 300.
    """

    # Configure the plot style
    sns.set_theme(style=style)

    # Setup for 1x3 grid plots
    _, axs = plt.subplots(1, 3, figsize=figsize)

    # Plot the data losses
    _lineplot_subplot(axs[0], model.data_loss_hist, 'Burgers Equation - Data Loss',
                      'Iteration', 'Error', 'Data Loss', y_scale='log')

    # Plot the physics losses
    _lineplot_subplot(axs[1], model.physics_loss_hist, 'Burgers Equation - Physics Loss',
                      'Iteration', 'Error', 'Physics Loss', y_scale='log')

    # Plot the total losses
    _lineplot_subplot(axs[2], model.loss_hist, 'Burgers Equation - Total Loss',
                      'Iteration', 'Error', 'Total Loss', y_scale='log')

    # Finalize layout
    sns.despine()
    plt.tight_layout()
    plt.legend()

    # Export the plot
    plt.savefig(path, dpi=dpi)
    plt.close()
