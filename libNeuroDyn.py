import numpy as np
import logging
from scipy import stats
from scipy.special import erfcx

logging.getLogger('NeuroDyn').addHandler(logging.NullHandler())

try:
    import numba
except:
    raise ImportError('Install numba. Available in pip, anaconda, etc.\nTested versions 0.13.0, 0.37.0')

if numba.__version__ not in ('0.13.0', '0.37.0'):
    logging.warning('Numba version %s not tested, it may have unexpected behavior.' % numba.__version__)

__author__ = 'Jose M. Esnaola Acebes'

""" Library containing functions to simulate Spiking Neuron's dynamics and Firing rate equations
    - Spiking neuron dynamics:
      + Current base: QIF, QIF + noise. INTEGRATION
      + Conductance base: QIF. INTEGRATION
    - Firing rate dynamics:
      + Conformal map: from FR to Kuramoto order parameter
      + Sigmoid functions for the transfer function in WC-type FR eqs.

"""

pi = np.pi


# -- Spiking neurons with euler integration.
# ------------------------------------------

# Deterministic QIF neuron for numba
@numba.jit()
def qifint_nf(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + tau * s_0[int(n / dn)])  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -d['v'][n]
    return d


@numba.jit()
def qifint_fr(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + s_0)  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -1.0 * d['v'][n]
    return d


@numba.jit()
def qifint_fr(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + s_0)  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -1.0 * d['v'][n]
    return d


# Deterministic QIF neuron with conductance based dynamics for numba
@numba.jit()
def qifint_cond(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, refr_tau, tau_peak, reversal, g):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is compute
    d when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the
     midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            # Euler integration
            d['v'][n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] - g * s_0 * (v[n] - reversal))
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -1.0 * d['v'][n]
    return d


# Deterministic QIF neuron with conductance based dynamics for numba
@numba.jit()
def qifint_cond2(v_exit_s1, v, exit0, eta_0, s_e, s_i, tiempo, number, dt, tau, vpeak, refr_tau, tau_peak, rev_e,
                 rev_i):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is compute
    d when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the
     midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            # Euler integration
            d['v'][n] = v[n] + (dt / tau) * (
                v[n] * v[n] + eta_0[n] - v[n] - s_e * (v[n] - rev_e) - s_i * (v[n] - rev_i))
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -1.0 * d['v'][n]
    return d


# Noisy QIF neuron for numba
@numba.jit()
def qifint_noise(v_exit_s1, v, exit0, eta_0, s_0, nois, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    d = v_exit_s1
    # These steps are necessary in order to use Numba (don't ask why ...)
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0 + tau * s_0[int(n / dn)]) + nois[
                n]  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t + refr_tau - (tau_peak - 1.0 / d['v'][n])
                d['s'][n] = 1
                d['v'][n] = -d['v'][n]
    return d


# IF neuron (single pop)
@numba.jit()
def ifint_fr(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, vreset, reversal, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (-v[n] + reversal + eta_0[n] + s_0)  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t * 1.0 + refr_tau
                d['s'][n] = 1
                d['v'][n] = vreset
    return d


# IF neuron (single pop)
@numba.jit()
def ifint_nf(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, vreset, reversal, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            d['v'][n] = v[n] + (dt / tau) * (-v[n] + reversal + eta_0[n] + s_0[int(n / dn)])  # Euler integration
            if d['v'][n] >= vpeak:
                d['t'][n] = t * 1.0 + refr_tau
                d['s'][n] = 1
                d['v'][n] = vreset
    return d


# EIF neuron (single pop)
@numba.jit()
def eifint_fr(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, vreset, reversal,
              refr_tau, tau_peak, sharp, rheo):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            # Euler integration
            d['v'][n] = v[n] + (dt / tau) * (-v[n] + reversal + sharp * np.exp((v[n] - rheo) / sharp) + eta_0[n] + s_0)
            if d['v'][n] >= vpeak:
                d['t'][n] = t * 1.0 + refr_tau
                d['s'][n] = 1
                d['v'][n] = vreset
                # if d['v'][n] <= -100:
                #     d['v'][n] = -100
    return d


# EIF neuron (neural field)
@numba.jit()
def eifint_nf(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, vreset, reversal,
              refr_tau, tau_peak, sharp, rheo):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d['s'][n] = 0
        if t >= exit0[n]:
            # Euler integration
            d['v'][n] = v[n] + (dt / tau) * (-v[n] + reversal + sharp * np.exp((v[n] - rheo) / sharp)
                                             + eta_0[n] + s_0[int(n / dn)])
            if d['v'][n] >= vpeak:
                d['t'][n] = t * 1.0 + refr_tau
                d['s'][n] = 1
                d['v'][n] = vreset
                # if d['v'][n] <= -100:
                #     d['v'][n] = -100
    return d


# - - Distributions
# -----------------
# Lorentz distribution
def lorentz(n, center, width):
    k = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n + 1.0)
    y = center + width * np.tan((np.pi / 2.0) * k)
    return y


# Gaussian distribution
def gauss(n, center, width):
    k = (np.arange(1, n + 1)) / (n + 1.0)
    y = center + width * stats.norm.ppf(k)
    return y


# Noise
def noise(length=100, disttype='g'):
    if disttype == 'g':
        return np.random.randn(length)


def conf_w_to_z(r, v):
    w = np.pi * np.array(r) + 1.0j * np.array(v)
    z = (1.0 - np.conjugate(w)) / (1.0 + np.conjugate(w))
    mod = np.abs(z)
    phase = np.angle(z)

    return mod, phase


def sigmoid(x):
    alpha = 1.5
    beta = 3.0
    i0 = -1.0
    return alpha / (1 + np.exp(-beta * (x - i0)))


def sigmoid_brunel_hakim(x):
    return 1 + np.tanh(x)


def sigmoid_qif(x, tau=1.0, delta=1.0, **kwargs):
    return (1.0 / (tau * pi * np.sqrt(2.0))) * np.sqrt(x + np.sqrt(x * x + delta * delta))


def sigmoid_lif(i0, tau=1.0, sigma=1.0, vr=-68.0, vth=-48.0, vrevers=-50.0, dt=10E-3):
    x1 = (vr - vrevers - i0) / sigma
    x2 = (vth - vrevers - i0) / sigma
    dx = (x2 - x1) / 2000.0
    x = np.arange(x1, x2 + dx, dx, dtype=np.float64)
    fx = erfcx(-x)
    t = tau * np.sqrt(pi) * np.sum(fx * dx) + dt * tau
    return 1.0 / t
