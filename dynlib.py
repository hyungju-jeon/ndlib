# %%
"""
library for Dynamical system
"""
from re import T
import torch
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.ndlib.misclib import Timer
from utils.ndlib.vislib import *
from utils.ndlib.misclib import get_cursor_pos

torch.set_default_dtype(torch.double)


# %% Dynamical system classes
class DynamicalSystem:
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


class LinearDynamics(DynamicalSystem):
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
            self.A @ self.x + MultivariateNormal(torch.zeros(self.n), self.Q).sample()
        )


class LimitCycle(DynamicalSystem):
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
        self.r += (self.r * (self.d - self.r**2)) * self.dt
        self.theta += (self.w + u) * self.dt

        self.x = (
            self.r * torch.tensor([torch.cos(self.theta), torch.sin(self.theta)])
            + MultivariateNormal(torch.zeros(self.n), self.Q).sample()
        )
        self.r = torch.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.theta = torch.atan2(self.x[1], self.x[0])


class TwoLimitCycle(DynamicalSystem):
    """
    Class representing two limit cycle dynamical system inherting from the dynamicalSystem class.
    Both limit cycles have the same frequency but the phase of only one of them will be perturbed.
    state :
        x = [r1, theta1, r2, theta2]
        r' = r(d - r^2)
        theta' = w
    """

    def __init__(self, ref_cycle, perturb_cycle, dt=None, y0=None, C=None, R=None):
        """
        Initialize a dynamical system.

        """
        x0 = torch.cat([ref_cycle.get_state(), perturb_cycle.get_state()])
        super().__init__(x0, dt)
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

    def get_phase_diff(self):
        return (self.reference.theta - self.perturb.theta) % (2 * np.pi)


# %% Observation model classes
class ObservationModel:
    def __init__(self):
        pass

    def get_observation(self):
        pass


class LinearObservation(ObservationModel):
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
        return self.C @ x + MultivariateNormal(torch.zeros(self.m), self.R).sample()


# %%
if __name__ == "__main__":
    obs_noise = 1e-4
    cycle_info = {
        "x0": torch.tensor([1.5, 0]),
        "d": 1,
        "w": 0.5,
        "Q": torch.tensor([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": 1e-2,
    }
    reference_cycle = LimitCycle(**cycle_info)
    perturb_cycle = LimitCycle(**cycle_info)
    twoC = TwoLimitCycle(reference_cycle, perturb_cycle, dt=1e-2)

    traj = np.zeros((200, 4))
    phase = np.zeros((200, 2))
    fig = plt.figure(figsize=(9, 3))
    ax_ref = fig.add_subplot(1, 3, 1)
    ax_perturb = fig.add_subplot(1, 3, 2)
    ax_phase = fig.add_subplot(1, 3, 3)
    plot_info = {"xlim": (-1.1, 1.1), "ylim": (-1.1, 1.1)}
    refTraj = BlitPlot(
        np.zeros((1, 2)),
        "trajectory",
        fig=fig,
        ax=ax_ref,
        **plot_info,
        title="Reference trajectory",
    )
    pertTraj = BlitPlot(
        np.zeros((1, 2)),
        "trajectory",
        fig=fig,
        ax=ax_perturb,
        **plot_info,
        title="Perturbed trajectory",
    )

    phaseTraj = BlitPlot(
        np.zeros((1, 2)),
        "trajectory",
        fig=fig,
        ax=ax_phase,
        **plot_info,
        title="Direction (Phase difference)",
    )

    for i in range(100000):
        pos = get_cursor_pos()
        if pos[0] > 0.25:
            twoC.update_state(pos[0])
        elif pos[0] < -0.25:
            twoC.update_state(pos[0])
        else:
            twoC.update_state(0)
        if i < 200:
            traj[i, :] = twoC.get_state()
            phase_diff = twoC.get_phase_diff()
            phase[i, :] = [np.cos(phase_diff), np.sin(phase_diff)]
        else:
            traj[0, :] = twoC.get_state()
            traj = np.roll(traj, -1, axis=0)
            phase_diff = twoC.get_phase_diff()
            phase[0, :] = [np.cos(phase_diff), np.sin(phase_diff)]
            phase = np.roll(phase, -1, axis=0)

            refTraj.refresh(traj[:, :2])
            pertTraj.refresh(traj[:, 2:])
            phaseTraj.refresh(np.vstack([phase, [0, 0]]))

            fig.canvas.flush_events()
