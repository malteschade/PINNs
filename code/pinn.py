#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Class module for the definition of the PINN.}

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
import tensorflow as tf

# -------- CLASSES --------
class PhysicsInformedNN:
    """
    Class for the definition of the PINN.
    """

    def __init__(self, x, x_f, u, layers, lb, ub, v):
        # Define instance variables
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        self.x = x[:, 0:1]
        self.t = x[:, 1:2]
        self.u = u

        self.x_f = x_f[:, 0:1]
        self.t_f = x_f[:, 1:2]

        self.layers = layers

        # Initialize Weights and Biases
        self.weights, self.biases = self.initialize_nn(layers)

        # Define parameters for the PINN
        self.lambda_1 = tf.Variable([v], dtype=tf.float64)

        # Initialize callback to save the loss at each iteration
        self.data_loss_hist = []
        self.physics_loss_hist = []
        self.loss_hist = []

    def initialize_nn(self, layers):
        """Initializes the weights and biases for the neural network.

        Args:
            layers (list[int]): The number of neurons in each layer.

        Returns:
            weights (list[tf.Variable]): The weights for the neural network.
            biases (list[tf.Variable]): The biases for the neural network.
        """

        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            w = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64))
            weights.append(w)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        """Initializes the weights using the Xavier initialization.

        Args:
            size ([int, int]): The size of the weights.

        Returns:
            tf.Variable: The initialized weights.
        """

        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim)).astype(np.float64)
        return tf.Variable(tf.random.normal(
            [in_dim, out_dim],stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    # Deep Neural Network (u_pred(x, t)==u(x, t))
    @tf.function
    def net_u(self, x, t):
        """The neural network for the solution.

        Args:
            x (np.ndarray): The x values.
            t (np.ndarray): The t values.

        Returns:
            tf.Tensor: The predicted solution for u.
        """

        h = 2.0 * (tf.concat([x, t], 1) - self.lb) / (self.ub - self.lb) - 1.0
        for w, b in zip(self.weights, self.biases):
            h = tf.tanh(tf.add(tf.matmul(h, w), b))
        return h

    # Physics Informed Neural Network (f(x, t)==0, 1D Burgers EQ)
    @tf.function
    def net_f(self, x, t):
        """The neural network for the physics-informed solution.

        Args:
            x (np.ndarray): The x values.
            t (np.ndarray): The t values.

        Returns:
            tf.Tensor: The predicted solution for f.
        """

        # Calculate the forward pass
        u = self.net_u(x, t)

        # Calculate the gradients with automatic differentiation
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        # Calculate the residual
        f = u_t + u * u_x - self.lambda_1 * u_xx
        return f

    # Callback to save the loss at each iteration
    def callback(self, losses):
        """Saves the loss at each iteration.

        Args:
            losses (list[float]): The data loss, physics loss, and overall loss.
        """

        self.data_loss_hist.append(losses[0])
        self.physics_loss_hist.append(losses[1])
        self.loss_hist.append(losses[2])

    # Training method
    def train(self, n_iter, learning_rate):
        """Trains the PINN.

        Args:
            n_iter (int): The number of iterations.
            learning_rate (float): The learning rate.
        """

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for it in range(n_iter):
            with tf.GradientTape() as tape:
                # Watch the weights and biases
                _ =((tape.watch(w), tape.watch(b)) for w, b in zip(self.weights, self.biases))

                # Predict the solution and the physics-informed neural network
                u_pred = self.net_u(self.x, self.t)
                f_pred = self.net_f(self.x_f, self.t_f)

                # Calculate the losses
                data_loss = tf.reduce_mean(tf.square(self.u - u_pred))
                physics_loss = tf.reduce_mean(tf.square(f_pred))
                loss = data_loss + physics_loss

            # Calculate the gradients and update the weights and biases
            gradients = tape.gradient(loss, self.weights + self.biases)
            optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

            # Save the loss at each iteration
            self.callback([data_loss.numpy(), physics_loss.numpy(), loss.numpy()])

            # Print the loss at each 1000th iteration
            if it % 1000 == 0:
                print(f"Iter: {it}, Data Loss: {data_loss.numpy():.5e}, " \
                      f"Physics Loss: {physics_loss.numpy():.5e}, Overall Loss: {loss.numpy():.5e}")

    # Prediction method
    def predict(self, x_star):
        """Predicts the solution for u and f.

        Args:
            x_star (np.ndarray): The x and t values.

        Returns:
            np.ndarray: The predicted solution for u.
            np.ndarray: The predicted solution for f.
        """

        u_star = self.net_u(x_star[:, 0:1], x_star[:, 1:2])
        f_star = self.net_f(x_star[:, 0:1], x_star[:, 1:2])
        return u_star.numpy(), f_star.numpy()
