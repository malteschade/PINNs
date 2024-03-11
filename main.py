#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Main module for running the Burger Equation forward PINN.}

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
# Built-in modules
from types import SimpleNamespace
import pathlib
import json

# Own modules
from code.pinn import PhysicsInformedNN
from code.functions import (read_data, flatten_data, seed_random, select_data,
                            define_collocation_points, calculate_error)
from utilities.plotting import plot_solution, plot_losses

# -------- CONSTANTS --------
CONFIG = 'config.json'

# -------- FUNCTIONS --------
def main() -> None:
    """
    Runs the Burger Equation forward PINN.
    """

    # Read the run configuration file
    path = pathlib.Path(__file__).parent / 'config' / CONFIG
    cfg = SimpleNamespace(**json.loads(path.read_text()))

    # Read the data
    x, t, exact = read_data(cfg.data_path)

    # Flatten the data
    x_star, u_star = flatten_data(x, t, exact)

    # Seed the random number generator
    seed_random(cfg.seed)

    # Select training data
    x_u_train, u_train = select_data(x_star, u_star, cfg.n_u, cfg.noise)

    # Select collocation points
    x_f_train = define_collocation_points(cfg.n_f, cfg.min_x, cfg.max_x,
                                          cfg.min_t, cfg.max_t)

    # Initialize the model
    model = PhysicsInformedNN(x_u_train, x_f_train, u_train, cfg.layers,
                              [cfg.min_x, cfg.min_t], [cfg.max_x, cfg.max_t],
                              cfg.v)

    # Train the model
    model.train(cfg.n_iter, cfg.learning_rate)

    # Predict the solution
    u_pred, _ = model.predict(x_star)
    error = calculate_error(u_pred, u_star)
    print(f'Prediction error: {error:.5e}')

    # Plot the results
    plot_solution(u_pred, exact, cfg.solution_plot_path)
    plot_losses(model, cfg.losses_plot_path)

# -------- SCRIPT --------
if __name__ == '__main__':
    main()
