# %%
"""
library for Dynamical system
"""
import numpy as np
import torch

torch.set_default_dtype(torch.double)


# %% Dynamical system classes (Numpy)
class AbstractDynamicalSystemNumpy:
    """
    class that incorporates all the dynamical systems
    """

    def __init__(self, x0, dt=1e-3):
        """
        Initialize a dynamical system.
        """
        self.dt = dt
        self.x = x0
        self.y = None
        if isinstance(x0, (int, float)):
            self.n = 1
        else:
            self.n = x0.shape[0]
        self.has_obs = False

    def set_state_dynamics(self):
        pass

    def set_state(self, x):
        """
        Set the state of the dynamical system.

        Parameters:
        - x0: torch tensor, the initial state of the dynamical system.

        Returns:
        - None
        """
        self.x = x

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

    def update_state(self):
        pass

    def update_observation(self):
        """
        Update the observation of the dynamical system.

        Returns:
        - None
        """
        pass

    def update(self):
        """
        Update and return the state and the observation of the dynamical system.

        Returns:
        - current_state: torch tensor, the current state of the dynamical system.
        - current_observation: torch tensor, the current observation of the dynamical system. (if exists)
        """
        if self.has_obs:
            return self.update_state(), self.update_observation()
        return self.update_state(), None

    def generate(self, t, update=False):
        """
        Generate the trajectory of the dynamical system.

        Parameters:
        - t: int, the number of time steps to generate.
        - update: bool, whether to update the dynamical system.

        Returns:
        - trajectory: torch tensor, the generated trajectory of the dynamical system.
        """
        trajectory = np.zeros((t, self.n))
        trajectory[0] = self.x

        observation = np.zeros((t, self.m))
        observation[0] = self.y

        x0 = self.get_state()
        y0 = self.get_observation()

        for i in range(1, t):
            self.update_state()
            trajectory[i] = self.get_state()
            observation[i] = self.y

        if not update:
            self.x = x0
            self.y = y0

        return trajectory, observation

    def generate_trajectory(self, t, u=None, update=False):
        """
        Generate the trajectory of the dynamical system.

        Parameters:
        - t: int, the number of time steps to generate.
        - update: bool, whether to update the dynamical system.

        Returns:
        - trajectory: torch tensor, the generated trajectory of the dynamical system.
        """
        trajectory = np.zeros((t, self.n))
        trajectory[0] = self.x

        x0 = self.get_state()

        for i in range(1, t):
            if u is not None:
                self.update_state(u[i, :])
            else:
                self.update_state()
            trajectory[i] = self.get_state()

        if not update:
            self.x = x0

        return trajectory

    def generate_observation(self, n, update=False):
        """
        Generate the observation of the dynamical system.

        Parameters:
        - n: int, the number of time steps to generate.
        - update: bool, whether to update the dynamical system.

        Returns:
        - observation: torch tensor, the generated observation of the dynamical system.
        """
        if not self.has_obs:
            raise ValueError("The dynamical system has no observation model.")

        observation = np.zeros((n, self.m))
        observation[0] = self.y

        x0 = self.get_state()
        y0 = self.get_observation()

        for i in range(1, n):
            self.update()
            observation[i] = self.y

        if not update:
            self.x = x0
            self.y = y0

        return observation

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        return self.x

    def get_observation(self):
        """
        Get the observation of the dynamical system.

        Returns:
        - y: torch tensor, the observation of the dynamical system.
        """
        return self.y


class LinearDynamicsNumpy(AbstractDynamicalSystemNumpy):
    """
    Class representing a linear dynamical system inherting from the dynamicalSystem class.
    state :
        x = Ax + v
        v ~ normal(0, Q)
    """

    def __init__(self, x0, A, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.A = A
        self.Q = Q

    def update_state(self):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.x = self.A @ self.x


class LimitCircleNumpy(AbstractDynamicalSystemNumpy):
    """
    Class representing a limit cycle dynamical system inherting from the dynamicalSystem class.
    state :
        x = [r, theta]
        r' = r(d - r^2)
        theta' = w
    """

    def __init__(self, x0, d, w, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.r = np.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.theta = np.arctan2(self.x[1], self.x[0])
        self.d = d
        self.w = w
        self.Q = Q

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        # If u is an array, perturbe the state x directly with u
        if isinstance(u, np.ndarray):
            # u has to have the same dimension as x
            if u.shape != self.x.shape:
                raise ValueError(
                    "Perturbation vector u must have the same dimension as the state x."
                )
            self.r += (self.r * (self.d - self.r) - u[1]) * self.dt
            self.r = np.fmax(self.r, 0.05)
            self.theta += (self.w + u[0]) * self.dt

            if self.Q is None:
                self.x = self.r * np.array([np.cos(self.theta), np.sin(self.theta)])
            else:
                self.x = self.r * np.array(
                    [np.cos(self.theta), np.sin(self.theta)]
                ) + np.random.multivariate_normal(np.zeros(self.n), self.Q)
            # self.x += u

        else:
            self.r += (self.r * (self.d - self.r)) * self.dt
            self.theta += (self.w + u) * self.dt

            if self.Q is None:
                self.x = self.r * np.array([np.cos(self.theta), np.sin(self.theta)]) + u
            else:
                self.x = self.r * np.array(
                    [np.cos(self.theta), np.sin(self.theta)]
                ) + np.random.multivariate_normal(np.zeros(self.n), self.Q)
        self.r = np.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.theta = np.arctan2(self.x[1], self.x[0])

        return


class TwoLimitCycleNumpy(AbstractDynamicalSystemNumpy):
    """
    Class representing two limit cycle dynamical system inherting from the dynamicalSystem class.
    Both limit cycles have the same frequency but the phase of only one of them will be perturbed.
    state :
        x = [r1, theta1, r2, theta2]
        r' = r(d - r^2)
        theta' = w
    """

    def __init__(
        self,
        ref_cycle: LimitCircleNumpy,
        perturb_cycle: LimitCircleNumpy,
        y0=None,
        C=None,
        R=None,
    ):
        """
        Initialize a dynamical system.

        """
        if ref_cycle.dt != perturb_cycle.dt:
            raise ValueError(
                "Reference cycle and perturb cycle must have the same time step (dt)."
            )

        x0 = np.concatenate([ref_cycle.get_state(), perturb_cycle.get_state()])
        super().__init__(x0, ref_cycle.dt)
        self.reference = ref_cycle
        self.perturb = perturb_cycle

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.reference.update_state()
        self.perturb.update_state(u)

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        return np.concatenate([self.reference.get_state(), self.perturb.get_state()])

    def update_w(self, w):
        """
        Update the frequency of the dynamical system.
        """
        self.reference.w = w
        self.perturb.w = w

    def get_phase_diff(self):
        return (self.reference.theta - self.perturb.theta) % (2 * np.pi)


class VanDerPolNumpy(AbstractDynamicalSystemNumpy):
    """
    Class representing a Van der Pol dynamical system inherting from the dynamicalSystem class.
    state :
        x = [x, y]
        x' = y
        y' = mu(1 - x^2)y - x
    """

    def __init__(self, x0, mu, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.mu = mu
        self.Q = Q

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.x = (
            self.x
            + self.dt
            * np.array(
                [self.x[1], self.mu * (1 - self.x[0] ** 2) * self.x[1] - self.x[0]]
            )
            + np.random.multivariate_normal(np.zeros(self.n), self.Q).sample()
        )


class RingLimitCycleNumpy(AbstractDynamicalSystemNumpy):
    """
    Class representing a torus dynamical system inherting from the dynamicalSystem class.
    state :
        x = [x, y, z]
        theta' = 0
        phi' = w1
        d = d
    """

    def __init__(self, x0, d_r, d_p, w_r, w_p, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        Args:
            x0 (torch.Tensor): The initial state of the system. [x, y, z]
            w (float): The value of w.
            Q (torch.Tensor): The value of Q.
            dt (float): The time step size.
            y0 (torch.Tensor, optional): The initial output of the system. Defaults to None.
            C (torch.Tensor, optional): The value of C. Defaults to None.
            R (torch.Tensor, optional): The value of R. Defaults to None.
        """
        super().__init__(x0, dt)
        r = np.sqrt(x0[0] ** 2 + x0[1] ** 2)

        self.reference = LimitCircleNumpy(
            x0=x0[:2] * (d_r / r), d=d_r, w=w_r, Q=None, dt=dt
        )
        self.perturb = LimitCircleNumpy(
            x0=np.array([r - d_r, x0[2]]), d=d_p, w=w_p, Q=None, dt=dt
        )

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        if isinstance(u, np.ndarray):
            # u has to have the same dimension as x
            if u.shape != self.x.shape:
                raise ValueError(
                    "Perturbation vector u must have the same dimension as the state x."
                )
            self.reference.update_state(u[:2])
            projected_u = np.dot(2 * (self.x[:2] + u[:2]), u[:2])
            self.perturb.update_state(np.array([projected_u, u[2]]))
        else:
            self.reference.update_state(0)
            self.perturb.update_state(0)

        self.x = self.get_state()
        return

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        x = self.perturb.get_state()
        r = self.reference.d + x[0]

        return np.array(
            [
                r * np.cos(self.reference.theta),
                r * np.sin(self.reference.theta),
                x[1],
            ]
        )


# %% Dynamical system classes (torch)
class AbstractDynamicalSystemTorch:
    """
    class that incorporates all the dynamical systems
    """

    def __init__(self, x0, dt=1e-3):
        """
        Initialize a dynamical system.
        """
        self.dt = dt
        self.x = x0.double()
        self.y = None
        self.n = x0.shape[0]
        self.has_obs = False

    def set_state_dynamics(self):
        pass

    def set_state(self, x):
        """
        Set the state of the dynamical system.

        Parameters:
        - x0: torch tensor, the initial state of the dynamical system.

        Returns:
        - None
        """
        self.x = x

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

    def update_state(self):
        pass

    def update_observation(self):
        """
        Update the observation of the dynamical system.

        Returns:
        - None
        """
        pass

    def update(self):
        """
        Update and return the state and the observation of the dynamical system.

        Returns:
        - current_state: torch tensor, the current state of the dynamical system.
        - current_observation: torch tensor, the current observation of the dynamical system. (if exists)
        """
        if self.has_obs:
            return self.update_state(), self.update_observation()
        return self.update_state(), None

    def generate_trajectory(self, t, update=False):
        """
        Generate the trajectory of the dynamical system.

        Parameters:
        - t: int, the number of time steps to generate.
        - update: bool, whether to update the dynamical system.

        Returns:
        - trajectory: torch tensor, the generated trajectory of the dynamical system.
        """
        trajectory = torch.zeros((t, self.n))
        trajectory[0] = self.x

        x0 = self.get_state()

        for i in range(1, t):
            self.update_state()
            trajectory[i] = self.get_state()

        if not update:
            self.x = x0

        return trajectory

    def generate_observation(self, n, update=False):
        """
        Generate the observation of the dynamical system.

        Parameters:
        - n: int, the number of time steps to generate.
        - update: bool, whether to update the dynamical system.

        Returns:
        - observation: torch tensor, the generated observation of the dynamical system.
        """
        if not self.has_obs:
            raise ValueError("The dynamical system has no observation model.")

        observation = torch.zeros((n, self.m))
        observation[0] = self.y

        x0 = self.get_state()
        y0 = self.get_observation()

        for i in range(1, n):
            self.update()
            observation[i] = self.y

        if not update:
            self.x = x0
            self.y = y0

        return observation

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        return self.x

    def get_observation(self):
        """
        Get the observation of the dynamical system.

        Returns:
        - y: torch tensor, the observation of the dynamical system.
        """
        return self.y


class LinearDynamicsTorch(AbstractDynamicalSystemTorch):
    """
    Class representing a linear dynamical system inherting from the dynamicalSystem class.
    state :
        x = Ax + v
        v ~ normal(0, Q)
    """

    def __init__(self, x0, A, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.A = A
        self.Q = Q

    def update_state(self):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.x = (
            self.A @ self.x
            + torch.distributions.MultivariateNormal(
                torch.zeros(self.n), self.Q
            ).sample()
        )


class LimitCircleTorch(AbstractDynamicalSystemTorch):
    """
    Class representing a limit cycle dynamical system inherting from the dynamicalSystem class.
    state :
        x = [r, theta]
        r' = r(d - r^2)
        theta' = w
    """

    def __init__(self, x0, d, w, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.r = torch.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.theta = torch.atan2(self.x[1], self.x[0])
        self.d = d
        self.w = w
        self.Q = Q

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        # If u is an array, perturbe the state x directly with u
        if isinstance(u, torch.Tensor):
            # u has to have the same dimension as x
            if u.shape != self.x.shape:
                raise ValueError(
                    "Perturbation vector u must have the same dimension as the state x."
                )
            self.r += (self.r * (self.d - self.r**2)) * self.dt
            self.theta += (self.w) * self.dt

            if self.Q is None:
                self.x = self.r * torch.tensor(
                    [torch.cos(self.theta), torch.sin(self.theta)]
                )
            else:
                self.x = (
                    self.r
                    * torch.tensor([torch.cos(self.theta), torch.sin(self.theta)])
                    + torch.distributions.MultivariateNormal(
                        torch.zeros(self.n), self.Q
                    ).sample()
                )
            self.x += u

        else:
            self.r += (self.r * (self.d - self.r**2)) * self.dt
            self.theta += (self.w + u) * self.dt

            if self.Q is None:
                self.x = (
                    self.r
                    * torch.tensor([torch.cos(self.theta), torch.sin(self.theta)])
                    + u
                )
            else:
                self.x = (
                    self.r
                    * torch.tensor([torch.cos(self.theta), torch.sin(self.theta)])
                    + torch.distributions.MultivariateNormal(
                        torch.zeros(self.n), self.Q
                    ).sample()
                )
        self.r = torch.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.theta = torch.atan2(self.x[1], self.x[0])

        return


class TwoLimitCycleTorch(AbstractDynamicalSystemTorch):
    """
    Class representing two limit cycle dynamical system inherting from the dynamicalSystem class.
    Both limit cycles have the same frequency but the phase of only one of them will be perturbed.
    state :
        x = [r1, theta1, r2, theta2]
        r' = r(d - r^2)
        theta' = w
    """

    def __init__(self, ref_cycle, perturb_cycle, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        if ref_cycle.dt != perturb_cycle.dt:
            raise ValueError(
                "Reference cycle and perturb cycle must have the same time step (dt)."
            )

        x0 = torch.cat([ref_cycle.get_state(), perturb_cycle.get_state()])
        super().__init__(x0, ref_cycle.dt)
        self.reference = ref_cycle
        self.perturb = perturb_cycle

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.reference.update_state()
        self.perturb.update_state(u)

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        return torch.cat([self.reference.get_state(), self.perturb.get_state()])

    def update_w(self, w):
        """
        Update the frequency of the dynamical system.
        """
        self.reference.w = w
        self.perturb.w = w

    def get_phase_diff(self):
        return (self.reference.theta - self.perturb.theta) % (2 * np.pi)


class VanDerPolTorch(AbstractDynamicalSystemTorch):
    """
    Class representing a Van der Pol dynamical system inherting from the dynamicalSystem class.
    state :
        x = [x, y]
        x' = y
        y' = mu(1 - x^2)y - x
    """

    def __init__(self, x0, mu, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        super().__init__(x0, dt)
        self.mu = mu
        self.Q = Q

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        self.x = (
            self.x
            + self.dt
            * torch.tensor(
                [self.x[1], self.mu * (1 - self.x[0] ** 2) * self.x[1] - self.x[0]]
            )
            + torch.distributions.MultivariateNormal(
                torch.zeros(self.n), self.Q
            ).sample()
        )


class RingLimitCycleTorch(AbstractDynamicalSystemTorch):
    """
    Class representing a torus dynamical system inherting from the dynamicalSystem class.
    state :
        x = [x, y, z]
        theta' = 0
        phi' = w1
        d = d
    """

    def __init__(self, x0, d_r, d_p, w_r, w_p, Q, dt, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        Args:
            x0 (torch.Tensor): The initial state of the system. [x, y, z]
            w (float): The value of w.
            Q (torch.Tensor): The value of Q.
            dt (float): The time step size.
            y0 (torch.Tensor, optional): The initial output of the system. Defaults to None.
            C (torch.Tensor, optional): The value of C. Defaults to None.
            R (torch.Tensor, optional): The value of R. Defaults to None.
        """
        super().__init__(x0, dt)
        r = torch.sqrt(x0[0] ** 2 + x0[1] ** 2)

        self.reference = LimitCircleNumpy(
            x0=x0[:2] * (d_r / r), d=d_r, w=w_r, Q=None, dt=dt
        )
        self.perturb = LimitCircleNumpy(
            x0=torch.tensor([r - d_r, x0[2]]), d=d_p, w=w_p, Q=None, dt=dt
        )

    def update_state(self, u=0):
        """
        Update the state of the dynamical system.

        Returns:
        - None
        """
        if isinstance(u, torch.Tensor):
            # u has to have the same dimension as x
            if u.shape != self.x.shape:
                raise ValueError(
                    "Perturbation vector u must have the same dimension as the state x."
                )
            self.reference.update_state(u[:2])
            projected_u = torch.dot(2 * (self.x[:2] + u[:2]), u[:2])
            self.perturb.update_state(torch.tensor([projected_u, u[2]]))
        else:
            self.reference.update_state(0)
            self.perturb.update_state(0)

        self.x = self.get_state()
        return

    def get_state(self):
        """
        Get the state of the dynamical system.

        Returns:
        - x: torch tensor, the state of the dynamical system.
        """
        x = self.perturb.get_state()
        r = self.reference.d + x[0]

        return torch.tensor(
            [
                r * torch.cos(self.reference.theta),
                r * torch.sin(self.reference.theta),
                x[1],
            ]
        )


# %% Functions to generate dynamical systems
def linear_dynamics(x0, A, Q, dt, y0=None, C=None, R=None):
    if isinstance(x0, np.ndarray):
        return LinearDynamicsNumpy(x0, A, Q, dt, y0, C, R)
    elif isinstance(x0, torch.Tensor):
        return LinearDynamicsTorch(x0, A, Q, dt, y0, C, R)
    else:
        raise ValueError("x0 must be a numpy array or a torch tensor.")


def limit_circle(x0, d, w, Q, dt, y0=None, C=None, R=None):
    if isinstance(x0, np.ndarray):
        return LimitCircleNumpy(x0, d, w, Q, dt, y0, C, R)
    elif isinstance(x0, torch.Tensor):
        return LimitCircleNumpy(x0, d, w, Q, dt, y0, C, R)
    else:
        raise ValueError("x0 must be a numpy array or a torch tensor.")


def two_limit_circle(ref_cycle, perturb_cycle, y0=None, C=None, R=None):
    if isinstance(ref_cycle.x, np.ndarray):
        return TwoLimitCycleNumpy(ref_cycle, perturb_cycle, y0, C, R)
    elif isinstance(ref_cycle.x, torch.Tensor):
        return TwoLimitCycleNumpy(ref_cycle, perturb_cycle, y0, C, R)
    else:
        raise ValueError("x0 must be a numpy array or a torch tensor.")


def van_der_pol_dyn(x0, mu, Q, dt, y0=None, C=None, R=None):
    if isinstance(x0, np.ndarray):
        return VanDerPolNumpy(x0, mu, Q, dt, y0, C, R)
    elif isinstance(x0, torch.Tensor):
        return VanDerPolTorch(x0, mu, Q, dt, y0, C, R)
    else:
        raise ValueError("x0 must be a numpy array or a torch tensor.")


def ring_limit_cycle(x0, d_r, d_p, w_r, w_p, Q, dt, y0=None, C=None, R=None):
    if isinstance(x0, np.ndarray):
        return RingLimitCycleNumpy(x0, d_r, d_p, w_r, w_p, Q, dt, y0, C, R)
    elif isinstance(x0, torch.Tensor):
        return RingLimitCycleTorch(x0, d_r, d_p, w_r, w_p, Q, dt, y0, C, R)
    else:
        raise ValueError("x0 must be a numpy array or a torch tensor.")


# %% Observation model classes
class AbstractObservationModel:
    def __init__(self):
        pass

    def get_observation(self):
        pass


class LinearObservation(AbstractObservationModel):
    def __init__(self, C, R):
        self.C = C
        self.R = R
        self.m = C.shape[0]

    def get_observation(self, x):
        """
        Generate the observation from the state of the dynamical system.

        Returns:
        - observation: torch tensor, the generated observation.
        """
        return (
            self.C @ x
            + torch.distributions.MultivariateNormal(
                torch.zeros(self.m), self.R
            ).sample()
        )


# %% Noise model classes
class AbstractNoiseModel:
    def __init__(self, Q):
        self.Q = Q
        pass

    def get_noise(self):
        pass


class GaussianNoiseNumpy(AbstractNoiseModel):
    def __init__(self, Q):
        super().__init__(Q)

    def get_noise(self):
        """
        Generate the noise.

        Returns:
        - noise: torch tensor, the generated noise.
        """
        return np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)


class GaussianNoiseTorch(AbstractNoiseModel):
    def __init__(self, Q):
        super().__init__(Q)

    def get_noise(self):
        """
        Generate the noise.

        Returns:
        - noise: torch tensor, the generated noise.
        """
        return torch.distributions.MultivariateNormal(
            torch.zeros(self.Q.shape[0]), self.Q
        ).sample()


# %% Main
if __name__ == "__main__":
    obs_noise = 1e-10
    cycle_info = {
        "x0": torch.tensor([1.5, 0]),
        "d": 1,
        "w": 1,
        "Q": torch.tensor([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": 1e-2,
    }
    torch_cycle_torch = limit_circle(**cycle_info)

    cycle_info_np = {
        "x0": np.array([1.5, 0]),
        "d": 1,
        "w": 1,
        "Q": np.array([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": 1e-2,
    }
    torch_cycle_NP = limit_circle(**cycle_info_np)

    torch_time = np.zeros(1000)
    np_time = np.zeros(1000)
    for i in range(1000):
        with Timer() as t:
            torch_cycle_torch.update_state()
        torch_time[i] = t.msecs
        with Timer() as t:
            torch_cycle_NP.update_state()
        np_time[i] = t.msecs

    plt.plot(torch_time, label="Torch")
    plt.plot(np_time, label="Numpy")
    print(f"Average time for torch: {torch_time.mean()} ms")
    print(f"Average time for numpy: {np_time.mean()} ms")
    plt.legend()
    plt.show()
