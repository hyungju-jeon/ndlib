# %%
"""
library for Dynamical system

Returns
-------
_type_
    _description_
"""

from matplotlib import pyplot as plt
from sympy import Li
from torch.distributions.multivariate_normal import MultivariateNormal
import torch


class LinearDynamics:
    """
    Class representing a linear dynamical system.
    observation :
        y = Cx + w
        w ~ normal(0, R)
    state :
        x = Ax + v
        v ~ normal(0, Q)
    """

    def __init__(self, A, Q, x0):
        """
        Initialize a dynamical system.

        """
        self.A = A
        self.Q = Q
        self.x = x0.double()
        self.n = self.A.shape[0]

    def set_observation_model(self, C, R, y0):
        """
        Set the observation model of the dynamical system.

        Parameters:
        - C: torch tensor, the observation matrix.
        - R: torch tensor, the covariance matrix of the observation noise.

        Returns:
        - None
        """
        self.C = C
        self.R = R
        self.y = y0
        self.m = self.C.shape[0]

    def set_state(self, x):
        """
        Set the state of the dynamical system.

        Parameters:
        - x0: torch tensor, the initial state of the dynamical system.

        Returns:
        - None
        """
        self.x = x

    def update_state(self):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.x = (
            self.A @ self.x + MultivariateNormal(torch.zeros(self.n), self.Q).sample()
        )

    def update_observation(self):
        """
        Update the observation of the dynamical system.

        Returns:
        - None
        """
        self.y = (
            self.C @ self.x + MultivariateNormal(torch.zeros(self.m), self.R).sample()
        )

    def update(self):
        """
        Update the state and the observation of the dynamical system.

        Returns:
        - None
        """
        self.update_state()
        self.update_observation()

    def generate_trajectory(self, n):
        """
        Generate the trajectory of the dynamical system.

        Parameters:
        - n: int, the number of time steps to generate.

        Returns:
        - trajectory: torch tensor, the generated trajectory of the dynamical system.
        """
        trajectory = torch.zeros((n, self.n))
        trajectory[0] = self.x

        for i in range(1, n):
            self.update_state()
            trajectory[i] = self.x

        return trajectory

    def generate_observation(self, n):
        """
        Generate the observation of the dynamical system.

        Parameters:
        - n: int, the number of time steps to generate.

        Returns:
        - observation: torch tensor, the generated observation of the dynamical system.
        """
        observation = torch.zeros((n, self.m))
        observation[0] = self.y

        for i in range(1, n):
            self.update()
            observation[i] = self.y

        return observation

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        return self.x


if __name__ == "__main__":
    # Create an instance of the DynamicalSystem class
    A = torch.tensor([[0.99, 0.5], [-0.5, 0.99]])
    C = torch.tensor([[1.0, 0.0]])
    Q = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    R = torch.tensor([[0.5]])
    x = torch.tensor([1.0, 10.0])
    y = torch.tensor([0.0])

    lin_ds = LinearDynamics(A, Q, x)
    lin_ds.set_observation_model(C, R, y)

    # Generate and print the trajectory
    n = 100
    trajectory = lin_ds.generate_trajectory(n)
    print("Generated Trajectory:")
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    # print(trajectory)

    # Generate and print the observation
    observation = lin_ds.generate_observation(n)
    print("Generated Observation:")
    print(observation)
