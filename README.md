# Partial Differential Equation Solver

A (partial) differential equation is an equation that relates one or more functions and their (partial) derivatives. Such equations play a prominent role in many disciplines including engineering, physics, economics, and biology.

This project was built based on [Physics Informed Deep Learning (Part I): Data-driven
Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/pdf/1711.10561.pdf?fbclid=IwAR10P9WY7MjntNJ3uvgqzfv8dk4bp9k2nHZh4bWGCS5ODZWFWjmOLg8vgNo).

The method is completely data-driven, since no analytical solution are required. The needed data for the ANN training are created during the process. During the training, the ANN minimizes the boundary condition loss function, applying a regularisation term related to the differential equation.
