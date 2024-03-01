import torch
import time
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.ndlib.misclib import Timer
from utils.ndlib.vislib import *
from utils.ndlib.dynlib import *
from utils.ndlib.misclib import get_cursor_pos

dt = 1e-3


def test_VanDerPol():
    """
    Test the Van der Pol oscillator.
    """
    obs_noise = 1e-3
    cycle_info = {
        "x0": torch.tensor([1.5, 0]),
        "mu": 1,
        "Q": torch.tensor([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": 1e-2,
    }
    vdp = VanDerPolTorch(**cycle_info)
    traj = np.zeros((200, 2))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)

    plot_info = {"xlim": (-5, 5), "ylim": (-5, 5)}
    plot_traj = BlitPlot(
        np.zeros((1, 2)),
        "trajectory",
        fig=fig,
        ax=ax,
        **plot_info,
        title="Vand der Pol trajectory",
    )
    for i in range(100000):
        vdp.update_state()
        if i < 200:
            traj[i, :] = vdp.get_state()
        else:
            traj[0, :] = vdp.get_state()
            traj = np.roll(traj, -1, axis=0)
            plot_traj.refresh(traj)

        fig.canvas.flush_events()


def test_TwoLimitCycle():
    obs_noise = 1e-6
    cycle_info = {
        "x0": np.array([1.5, 0]),
        "d": 1,
        "w": 2 * np.pi * 2,
        "Q": np.array([[obs_noise, 0.0], [0.0, obs_noise]]),
        "dt": dt,
    }
    reference_cycle = LimitCircleNumpy(**cycle_info)
    perturb_cycle = LimitCircleNumpy(**cycle_info)
    twoC = TwoLimitCycleNumpy(reference_cycle, perturb_cycle)

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
        time.sleep(dt)


def test_RingLimitCycle():
    ring = RingLimitCycleTorch(torch.tensor([5, 0, 1]), 5, 1, 5, 200, 1e-4, 1e-4)

    trajectory = np.zeros((200, 3))

    fig = plt.figure(figsize=(9, 3))
    ax_ref = fig.add_subplot(1, 3, 1, projection="3d")
    plot_info = {"xlim": (-6, 6), "ylim": (-6, 6), "zlim": (-2, 2)}
    refTraj = BlitPlot(
        np.zeros((1, 4)),
        "trajectory3d",
        fig=fig,
        ax=ax_ref,
        **plot_info,
        title="Reference trajectory",
    )

    for i in range(100000):
        ring.update_state(0)
        if i < 200:
            trajectory[i, :] = ring.get_state()
        else:
            trajectory[0, :] = ring.get_state()
            trajectory = np.roll(trajectory, -1, axis=0)
            refTraj.refresh(trajectory[:, :])
            fig.canvas.flush_events()


if __name__ == "__main__":
    test_TwoLimitCycle()
    # test_VanDerPol()
    # test_RingLimitCycle()
