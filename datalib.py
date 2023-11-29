# %%
import numpy as np
import torch
import gpytorch
import h5py


def generate_GP_data(t_max=10, dt=0.01, d_latent=3, l=1, normalize=True, fname=None):
    """
    Generate Gaussian Process (GP) data.

    Args:
        t_max (float): Maximum time value.
        dt (float): Time step size.
        d_latent (int): Dimensionality of the latent trajectory.
        l (float): Lengthscale parameter for the RBF kernel.
        normalize (bool): Flag indicating whether to normalize the latent trajectory.
        fname (str): File name to save the latent trajectory (optional).

    Returns:
        torch.Tensor: The generated latent trajectory.
    """

    nT = int(t_max / dt)
    index_points = torch.linspace(0, t_max, nT).unsqueeze(-1)

    # Define a kernel with default parameters.
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = l * dt
    gp = gpytorch.distributions.MultivariateNormal(
        torch.zeros(nT), kernel(index_points)
    )

    samples = gp.sample(torch.Size([d_latent]))
    latentTraj = samples.T

    if normalize:
        latentTraj -= torch.mean(latentTraj, axis=0)
        latentTraj = latentTraj / torch.std(latentTraj, axis=0)  # Normalize

    if fname is not None:
        hf = h5py.File(fname, "w")
        hf["latentTraj"] = latentTraj
        hf.close()

    return latentTraj
