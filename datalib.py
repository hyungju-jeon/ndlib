# %%
import numpy as np
import torch
import gpytorch
import h5py


def generate_GP_data(
    t_max=10, dt=0.01, d_latent=3, lengthscale=1, normalize=True, fname=None
):
    """
    Generate Gaussian Process (GP) data.

    Args:
        t_max (float): Maximum time value.
        dt (float): Time step size.
        d_latent (int): Dimensionality of the latent space.
        lengthscale (float): Lengthscale parameter for the RBF kernel.
        normalize (bool): Flag indicating whether to normalize the generated data.
        fname (str): File name to save latent trajectory.

    Returns:
        numpy.ndarray: Generated latent trajectory data.
    """
    nT = int(t_max / dt)
    index_points = torch.linspace(0, t_max, nT).unsqueeze(-1)

    # Define a kernel with default parameters.
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = lengthscale * dt
    gp = gpytorch.distributions.MultivariateNormal(
        torch.zeros(nT), kernel(index_points)
    )

    samples = gp.sample(torch.Size([d_latent]))
    latentTraj = samples.numpy().T

    if normalize:
        latentTraj -= np.mean(latentTraj, axis=0)
        latentTraj = latentTraj / np.std(latentTraj, axis=0)  # Normalize

    if fname is not None:
        hf = h5py.File(fname, "r")
        latentTraj = hf["latentTraj"][:]
        hf.close()

    return latentTraj
