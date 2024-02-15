# Simulate a dynamical system as the latent dynamics of simulated neural data

# Note: All of our time series will be in tall and skinny format (see numpy_speedup.ipynb)

import numpy as np
import warnings
from scipy.linalg import lstsq


class AbstractDynamicalSystem:
    def __init__(self, dim, params):
        self.dim = dim
        self.params = params

    def f(self, x):
        # dx/dt = f(x)
        raise NotImplementedError

    def simulate(self, dt, T, x0):
        return self.simulateWithAdditiveInput(dt, T, x0, np.zeros((T, 2)))

    def simulateWithAdditiveInput(self, dt, T, x0, u):
        assert x0.shape[0] == self.dim
        assert dt > 0
        assert T > 0

        states = np.zeros((T, 2))
        states[0, :] = x0

        for kt in range(1, T):  # Euler-Maruyama integration
            states[kt, :] = states[kt-1, :] + dt * self.f(states[kt-1, :])
            states[kt, :] = states[kt, :] + u[kt, :]

        return states

    def simulateWithAdditiveNoise(self, dt, T, x0, sigma):
        stateNoise = np.random.randn(T, 2) * sigma / np.sqrt(dt)
        states = self.simulateWithAdditiveInput(dt, T, x0, stateNoise)

        return states, stateNoise


class VanDerPol(AbstractDynamicalSystem):
    """ Van der Pol oscillator """

    def __init__(self, params={'mu': 1.0}):
        dim = 2
        super().__init__(dim, params)

    def f(self, x):
        mu = self.params['mu']

        dx = np.array([x[1], mu * (1 - x[0]**2) * x[1] - x[0]])
        return dx


def computeFiringRate(x, C, b):
    """ Compute the firing rate of a log-linear Poisson neuron model """
    return np.exp(x @ C + b)


def SNR_dB_logLinearPoissonNeurons_fastBound(firing_rates, C):
    """ Evaluate the SNR of observations using Fisher Information Matrix of a log-linear Poisson neuron model.
    Signal is defined to be the latent states x.
    Each neuron and time points are assumed to be independent. This is not a realistic assumption,
    but it gives an upper bound of the SNR. Each sample/time point is assumed to be independent,
    and the expected SNR per sample is returned.
    """
    assert firing_rates.shape[1] == C.shape[1]
    SNR_ub = 10 * np.log10(np.mean(firing_rates @ (C ** 2).T, axis=0))
    return SNR_ub


def _C_to_U2E2(C):
    """Intermediate vector needed for the super efficient computation of upper bound of SNR using Fisher information for log-linear-Poisson model.

    Parameters
    ----------
    C: (N x L) ndarray
        Loading matrix for N neurons and L latent dimensions. N >= L

    Returns
    -------
    U2E2: (N x 1) ndarray
        a vector when taken inner product with the inverse firing rate vector gives the upper bound of SNR

    References
    ----------
    Let us take the SVD of $\vC = \vU \vE \vV\trp$. The trace of the inverse Fisher information matrix is given by
    $$ C^T \diag(\lambda) C =
            \left(
                \sum_{k} \lambda_{k}^{-1} \sum_i U_{k,i}^2 E_{i,i}^{-2}
            \right)
    $$
    SNR is inversely proportional to the expected value of this quantity.
    """
    assert C.shape[0] >= C.shape[1], 'C must have more rows than columns'
    U, E, _ = np.linalg.svd(C, full_matrices=False)

    return (np.square(U) @ (E ** -2))


def SNR_dB_logLinearPoissonNeurons_wInv(x, C, b):
    """Tighter bound for the SNR, but a slower implementation"""
    firing_rates = computeFiringRate(x, C, b)
    # slow implementation... what's bsxfun in numpy?
    FIM_e = np.mean([C @ np.diag(firing_rates[t, :]) @
                    C.T for t in range(x.shape[0])], axis=0)
    CR_var = np.linalg.inv(FIM_e)  # this is numerically fishy
    SNR = 10 * np.log10(1 / CR_var.diagonal())
    return SNR, FIM_e


def updateBiasToMatchTargetFiringRate(currentRatePerBin, currentBias, targetRatePerBin=0.05):
    # easy to find the bias to compensate for the change in firing rate due to changing C
    assert currentBias.size == currentRatePerBin.size
    return currentBias + np.log(targetRatePerBin/currentRatePerBin)


def computeSNR(latentTraj, C, b, targetRatePerBin):
    # note that the latentTraj is assumed to be of variance 1
    firing_rates = computeFiringRate(latentTraj, C, b)
    b = updateBiasToMatchTargetFiringRate(
        np.mean(firing_rates, axis=0), b, targetRatePerBin=targetRatePerBin)
    firing_rates = computeFiringRate(latentTraj, C, b)

    # U2E2 = _C_to_U2E2(C.T)
    # SNR = -10 * np.log10(np.mean(firing_rates**-1 @ U2E2))

    SNR = 0
    for i, firing_rate in enumerate(firing_rates):
        SNR += np.trace(np.linalg.inv(C @ np.diag(firing_rate) @ C.T))
    SNR = SNR / firing_rates.shape[0]
    SNR = 10 * np.log10(C.shape[0]/SNR)

    return SNR, b


def computeSNR_2(latentTraj, C, b, targetRatePerBin):
    # note that the latentTraj is assumed to be of variance 1
    firing_rates = computeFiringRate(latentTraj, C, b)
    b = updateBiasToMatchTargetFiringRate(
        np.mean(firing_rates, axis=0), b, targetRatePerBin=targetRatePerBin)
    firing_rates = computeFiringRate(latentTraj, C, b)

    SNR = 0
    for i, firing_rate in enumerate(firing_rates):
        SNR += (
            np.exp(-np.sum(np.log(C**2 @ firing_rate)) / latentTraj.shape[1])
            * latentTraj.shape[1]
        )
    SNR = SNR / firing_rates.shape[0]
    SNR = 10 * np.log10(C.shape[0]/SNR)

    return SNR, b


def scaleCforTargetSNR(latentTraj, C, b, targetRatePerBin, targetSNR, SNR_method):
    maxGain = 1.0
    for _ in range(20):
        SNR, _ = SNR_method(latentTraj, C * maxGain, b, targetRatePerBin)
        if SNR > targetSNR:
            break
        else:
            maxGain = maxGain * 1.5

    minGain = 0.5
    for _ in range(20):
        SNR, _ = SNR_method(latentTraj, C * minGain, b, targetRatePerBin)
        if SNR > targetSNR:
            minGain = minGain * 0.5
        else:
            break

    #  start the bisection search for the target SNR
    for _ in range(40):
        gain = (maxGain + minGain) / 2
        SNR, _ = SNR_method(latentTraj, C * gain, b, targetRatePerBin)
        if SNR > targetSNR:
            maxGain = gain
        elif SNR <= targetSNR:
            minGain = gain

    SNR, b = SNR_method(latentTraj, C * gain, b, targetRatePerBin)

    if (SNR - targetSNR) > 0.1 * np.abs(targetSNR):
        print(
            f"Warning: SNR reached is way greater than the target SNR {targetSNR}. SNR =", SNR)
    if (SNR - targetSNR) < -0.1 * np.abs(targetSNR):
        print(
            f"Warning: SNR reached is less than the target SNR {targetSNR}. SNR =", SNR)

    return (C * gain), b, SNR


def scaleCforTargetSNR_dim(latentTraj, C, b, targetRatePerBin, targetSNR):
    maxGain = 1.0
    for _ in range(20):
        SNR, _ = computeSNR(latentTraj, C * maxGain, b, targetRatePerBin)
        if all(SNR > targetSNR):
            break
        elif any(SNR < targetSNR):
            maxGain = maxGain * 1.5

    minGain = 0.5
    for _ in range(20):
        SNR, _ = computeSNR(latentTraj, C * minGain, b, targetRatePerBin)
        if any(SNR > targetSNR):
            minGain = minGain * 0.5
        elif any(SNR < targetSNR):
            break

    #  start the bisection search for the target SNR
    for _ in range(20):
        gain = (maxGain + minGain) / 2
        SNR, _ = computeSNR(latentTraj, C * gain, b, targetRatePerBin)
        if all(SNR > targetSNR):
            maxGain = gain
        elif any(SNR <= targetSNR):
            minGain = gain

    SNR, b = computeSNR(latentTraj, C * gain, b, targetRatePerBin)

    if any(SNR < 0.9 * targetSNR):
        print('Warning: SNR reached is less than the target SNR. SNR =', SNR)

    return (C * gain), b, SNR


def autoGeneratePoissonObservations(latentTraj, C=None, dNeurons=100, targetRatePerBin=0.01, pCoherence=0.5, pSparsity=0.1, targetSNR=10.0, SNR_method=computeSNR):
    assert pSparsity >= 0, 'pSparsity must be between 0 and 1'
    assert pSparsity <= 1, 'pSparsity must be between 0 and 1'
    assert dNeurons > 0, 'dNeurons must be positive'
    assert targetRatePerBin > 0, 'targetRatePerBin must be positive'
    if not np.all(np.isclose(np.std(latentTraj, axis=0), 1)):
        print('WARNING: latent trajectory must have unit variance. Normalizing...')
        latentTraj = latentTraj / np.std(latentTraj, axis=0)

    dLatent = latentTraj.shape[1]
    C = generate_random_loading_matrix(
        dLatent, dNeurons, pCoherence, pSparsity, C=C)

    b = 1.0 * np.random.rand(1, dNeurons) - np.log(targetRatePerBin)
    C, b, SNR = scaleCforTargetSNR(
        latentTraj, C, b, targetRatePerBin, targetSNR=targetSNR, SNR_method=SNR_method)
    firing_rates = computeFiringRate(latentTraj, C, b)

    observations = np.random.poisson(firing_rates)
    # observations[observations > 1] = 1  # binarize

    return observations, C, b, firing_rates, SNR


def compute_mutual_coherence(C):
    C_normalized = C / np.linalg.norm(C, axis=0)
    CC = C_normalized.T @ C_normalized

    return np.max(np.abs(CC - np.diag(np.diag(CC))))


def generate_random_loading_matrix(dLatent, dNeurons, pCoherence, pSparsity=0, C=None):
    # Constructing Low Mutual Coherence Matrix
    # via Direct Mutual Coherence Minimization (DMCM)
    # Lu, Canyi, Huan Li, and Zhouchen Lin. "Optimized projections for compressed sensing via direct mutual coherence minimization." Signal Processing 151 (2018): 45-55.
    if C is None:
        C = np.random.randn(dLatent, dNeurons)
        C = C * (np.random.rand(dLatent, dNeurons) > pSparsity)
        C /= np.linalg.norm(C, axis=0)

    T = 15
    K = 1000
    rho = 0.5
    eta = 1.1
    lbda = 0.9
    alpha = lbda * rho

    coh = compute_mutual_coherence(C)
    for _ in range(T):
        for _ in range(K):
            coh = compute_mutual_coherence(C)
            if coh < pCoherence:
                C /= np.linalg.norm(C, axis=0)
                return C

            VV = (C.T@C - np.eye(dNeurons))/rho
            v = euclidean_proj_l1ball(VV.flatten(), s=1)
            V = np.reshape(v, (dNeurons, dNeurons))

            MM = C - alpha * C @ (V + V.T)
            C = MM / np.linalg.norm(MM, axis=0)
        rho = rho / eta
        alpha = lbda * rho

    if coh >= pCoherence:
        warnings.warn(
            f'target Coherence {pCoherence} not reached, Current Coherence {coh}')

    C /= np.linalg.norm(C, axis=0)
    return C


def decode_latents_MLE_fast(observations, C, b):
    """
    Closed-form decoding of the latent states from spike trains using approximate MLE formula.
    It approximates the exponential function of the latent state as a linear function around the origin.
    """
    Ymod = (observations * np.exp(-b) - 1)
    Xamle = lstsq(C.T, Ymod.T)[0].T

    return Xamle


def ppLL_normalized(y, rate, lograte=[]):
    """
    Point process Log-likelihood in units of nats/bin.
    Larger log-likelihood indicates a better model of the observed data.
    Thanks to the normalization by number of spikes (events), it is independent of the bin size, though larger bin size limits the maximum value achievable.

    We use the Poisson likelihoods in discrete time:
    $ LL = \sum_t \left[ y(t) \log(\lambda(t)) - \lambda(t) - \log(y(t)!) \right] $

    relative to the null model (only mean firing rate $\bar{\lambda}$)
    $ LL0 = \sum_t \left[ y(t) \log(\bar{\lambda}) - \bar{\lambda} - \log(y(t)!) \right] $

    The difference normalized by the number of time bins is returned:
    $ LLn = (LL - LL0) / \text{number of bins} $

    Parameters
    ----------
    y: ndarray
        Array of spike counts (integer or float) per bin.

    rate: ndarray
        Per bin firing rate, non-negative values

    lograte: ndarray (optional)
        natural logarithm of the rate argument. Often precomputed for speed.

    Returns
    -------
    float
        Log-likelihood per bin minus the null model likelihood

    Note
    ----
        Other common units:
          - multiply by `np.log2(np.e)` to convert to bits/bin (or divide by `np.log(2)`)
          - multiply by mean spike count per bin to convert to nats/spk (as long as you have non-zero number of spikes)
          - divide by bin size in seconds to convert to nats/sec

    Raises
    ------
        ValueError for infinite firing rate.

    References
    ----------
    ..[1] Pillow, J. W., Shlens, J., Paninski, L., Sher, A., Litke, A. M., Chichilnisky, E. J., &
          Simoncelli, E. P. (2008).  Spatio-temporal correlations and visual signalling in a complete
          neuronal population. Nature, 454(7207), 995-999.

    Examples
    --------
    >>> import numpy as np
    >>> ppLL_normalized(np.array([0, 0, 0, 1, 0, 1]), np.array([0.5, 0.1, 0.01, 1.2, 0.1, 2.2]))
    """
    assert np.all(rate >= 0), 'firing rate must be non-negative'
    assert np.all(y >= 0), 'number of spikes must be natural numbers'

    if np.any(np.isinf(rate)):
        raise ValueError(
            'Firing rate per bin is Inf. You may have a bad initial parameters or optimization.')

    if np.size(lograte) == 0:
        # eps = np.finfo(rate.dtype).eps # smallest gap for floating point at 1.0
        # rate = np.where(rate > eps, rate, eps) # put lograte for 0 firing rate in numerical range
        # suppress the divide by zero in log(0) warning
        with np.errstate(divide='ignore', invalid='ignore'):
            lograte = np.log(rate)

    if np.any(np.isinf(lograte)):
        lograte = np.where(np.logical_and(y == 0, np.isinf(
            lograte)), 0, lograte)  # 0 times -inf = 0

    assert lograte.shape == y.shape == rate.shape, 'provide firing rate per bin'

    nSpk = np.sum(y, axis=0)  # number of spikes
    nT = y.shape[0]           # number of time bins
    mfr = nSpk / nT           # mean number of spikes per bin

    # CORE computation:
    # return (np.sum(y * lograte - rate, axis=0) / nT) - (np.sum(y * np.log(mfr) - mfr, axis=0) / nT)

    # suppress the divide by zero in log(0) warning
    with np.errstate(divide='ignore', invalid='ignore'):
        LL0 = mfr * np.log(np.where(mfr == 0, 1., mfr)) - mfr

    return np.sum(y * lograte - rate, axis=0) / nT - LL0


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def generate_observations(latentTraj, params, C=None):
    observations, C, b, firing_rate_per_bin, SNR = autoGeneratePoissonObservations(
        latentTraj[:, :params["dLatent"]],
        dNeurons=params["dNeurons"],
        targetRatePerBin=params["targetRatePerBin"],
        pCoherence=params["pCoherence"],
        pSparsity=params["pSparsity"],
        targetSNR=params["targetSNR"],
        SNR_method=computeSNR,
        C=C
    )
    return observations, C, b, firing_rate_per_bin, SNR


if __name__ == "__main__":
    vdp = VanDerPol(params={'mu': 1.0})
    x = vdp.simulate(dt=0.01, T=1000, x0=np.array([1.0, 1.0]))
    y, C, b, SNR = autoGeneratePoissonObservations(x, targetRatePerBin=0.1)

    print(C)
    print(b)
    print(SNR)
    print(np.mean(y, axis=0))
    print(np.max(y, axis=0))
