#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Setup script for the Burger Equation forward PINN.}

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
from setuptools import setup, find_packages

# -------- SETUP --------
with open('requirements.txt', encoding='utf8') as f:
    required = f.read().splitlines()

setup(
    name='pinn',
    version='1.0.0',
    packages=find_packages(),
    install_requires=required,
    author='Malte Leander Schade',
    author_email='mail@malteschade.com',
    description='',
    url='https://github.com/malteschade/PINNs',
    license='MIT License',
    status='Completed',
)
