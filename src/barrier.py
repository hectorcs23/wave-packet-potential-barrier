"""
barrier.py
==========

Rectangular potential barrier: stationary scattering states, transmission and
reflection coefficients, and the Gaussian wave packet built on top of them.

Units: hbar = m = 1.

    V(x) = V0   for 0 <= x <= a,   0 elsewhere

An incoming plane wave of unit amplitude from the left gives

    region I   (x < 0)      e^{ikx} + B e^{-ikx}
    region II  (0..a)       C e^{iqx}  + D e^{-iqx}     (E > V0,  q = sqrt(2(E-V0)))
                            C e^{kx}   + D e^{-kx}      (E < V0,  kappa = sqrt(2(V0-E)))
    region III (x > a)      F e^{ikx}

with T = |F|^2 and R = |B|^2.

The packet is

    Psi(x,t) = (2*pi)^{-1/2} \\int phi(k) psi_k(x) e^{-i k^2 t / 2} dk,
    phi(k)   = (2b/pi)^{1/4} / sqrt(2b) * e^{i x0 (l-k)} * e^{-(l-k)^2 / (4b)}

so that <k> = l, var(k) = b, <E> = (l^2 + b)/2 and var(E) = 2 <E> b - b^2 / 2.
Inverting the last two relations gives b and l from a requested <E> and var(E),
which is how the parameters of the simulation are set.
"""
from __future__ import annotations

import numpy as np

HBAR = 1.0
MASS = 1.0


# --------------------------------------------------------------- parameters
def packet_parameters(E_mean: float, E_var: float):
    """(b, l) = (var of k, central k) reproducing a requested mean energy and variance."""
    b = MASS / HBAR ** 2 * (2 * E_mean - np.sqrt(4 * E_mean ** 2 - 2 * E_var))
    l = np.sqrt(E_mean * 2 * MASS / HBAR ** 2 - b)
    return float(b), float(l)


def phi_k(ks, b, l, x0):
    """Momentum-space amplitude of the packet (peaked at l, variance b)."""
    norm = (2 * b / np.pi) ** 0.25 / np.sqrt(2 * b)
    return norm * np.exp(1j * x0 * (l - ks)) * np.exp(-(l - ks) ** 2 / (4.0 * b))


# ------------------------------------------------------- scattering solution
def coefficients(ks, V0, a):
    """Closed-form B, C, D, F for an array of wave numbers (both energy regimes)."""
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    E = (HBAR * ks) ** 2 / (2 * MASS)
    B = np.zeros_like(ks, dtype=complex)
    C = np.zeros_like(B)
    D = np.zeros_like(B)
    F = np.zeros_like(B)

    above = E > V0
    if np.any(above):
        k = ks[above]
        q = np.sqrt(2 * MASS * (E[above] - V0)) / HBAR
        den = (k + q) ** 2 - (q - k) ** 2 * np.exp(2j * q * a)
        B[above] = (k ** 2 - q ** 2) * (1 - np.exp(2j * q * a)) / den
        C[above] = 2 * k * (k + q) / den
        D[above] = 2 * k * (q - k) * np.exp(2j * q * a) / den
        F[above] = 4 * k * q * np.exp(1j * (q - k) * a) / den

    below = ~above
    if np.any(below):
        k = ks[below]
        kap = np.sqrt(2 * MASS * (V0 - E[below])) / HBAR
        den = (k + 1j * kap) ** 2 * np.exp(2 * kap * a) - (k - 1j * kap) ** 2
        B[below] = (k ** 2 + kap ** 2) * (np.exp(2 * kap * a) - 1) / den
        C[below] = -2 * k * (k - 1j * kap) / den
        D[below] = 2 * k * (k + 1j * kap) * np.exp(2 * kap * a) / den
        F[below] = 4j * k * kap * np.exp(kap * a) * np.exp(-1j * k * a) / den

    return B, C, D, F


def transmission(ks, V0, a):
    """T(k) in closed form — the textbook expressions, independent of `coefficients`."""
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    E = (HBAR * ks) ** 2 / (2 * MASS)
    T = np.empty_like(ks, dtype=float)

    above = E > V0
    k = ks[above]
    q = np.sqrt(2 * MASS * (E[above] - V0)) / HBAR
    T[above] = 1.0 / (1.0 + (k ** 2 - q ** 2) ** 2 * np.sin(q * a) ** 2 / (4 * k ** 2 * q ** 2))

    below = ~above
    k = ks[below]
    kap = np.sqrt(2 * MASS * (V0 - E[below])) / HBAR
    T[below] = 1.0 / (1.0 + (k ** 2 + kap ** 2) ** 2 * np.sinh(kap * a) ** 2 / (4 * k ** 2 * kap ** 2))
    return T


def continuity_residuals(k, V0, a):
    """
    |psi_I - psi_II| and |psi'_I - psi'_II| at x = 0, and the same at x = a,
    evaluated analytically from the coefficients (no finite differences), so the
    residual is limited only by floating point.

    Returns (d_psi_0, d_dpsi_0, d_psi_a, d_dpsi_a).
    """
    E = (HBAR * k) ** 2 / (2 * MASS)
    B, C, D, F = (c[0] for c in coefficients([k], V0, a))

    psi_I, dpsi_I = 1 + B, 1j * k * (1 - B)                       # at x = 0
    psi_III = F * np.exp(1j * k * a)                              # at x = a
    dpsi_III = 1j * k * psi_III

    if E > V0:
        q = np.sqrt(2 * MASS * (E - V0)) / HBAR
        psi_II0, dpsi_II0 = C + D, 1j * q * (C - D)
        ea, eb = np.exp(1j * q * a), np.exp(-1j * q * a)
        psi_IIa, dpsi_IIa = C * ea + D * eb, 1j * q * (C * ea - D * eb)
    else:
        kap = np.sqrt(2 * MASS * (V0 - E)) / HBAR
        psi_II0, dpsi_II0 = C + D, kap * (C - D)
        ea, eb = np.exp(kap * a), np.exp(-kap * a)
        psi_IIa, dpsi_IIa = C * ea + D * eb, kap * (C * ea - D * eb)

    return (abs(psi_I - psi_II0), abs(dpsi_I - dpsi_II0),
            abs(psi_IIa - psi_III), abs(dpsi_IIa - dpsi_III))


def psi_matrix(xs, ks, V0, a):
    """psi_k(x) for every (x, k) pair. Shape (len(xs), len(ks))."""
    xs = np.asarray(xs, dtype=float)
    ks = np.asarray(ks, dtype=float)
    E = (HBAR * ks) ** 2 / (2 * MASS)
    B, C, D, F = coefficients(ks, V0, a)

    psi = np.empty((xs.size, ks.size), dtype=complex)
    left, mid, right = xs < 0, (xs >= 0) & (xs <= a), xs > a

    if np.any(left):
        xl = xs[left][:, None]
        psi[left] = np.exp(1j * ks * xl) + B * np.exp(-1j * ks * xl)

    if np.any(mid):
        xm = xs[mid][:, None]
        above = E > V0
        q = np.where(above, np.sqrt(2 * MASS * np.abs(E - V0)) / HBAR, 0.0)
        kap = np.where(above, 0.0, np.sqrt(2 * MASS * np.abs(V0 - E)) / HBAR)
        osc = C * np.exp(1j * q * xm) + D * np.exp(-1j * q * xm)
        evan = C * np.exp(kap * xm) + D * np.exp(-kap * xm)
        psi[mid] = np.where(above, osc, evan)

    if np.any(right):
        xr = xs[right][:, None]
        psi[right] = F * np.exp(1j * ks * xr)

    return psi


# ------------------------------------------------------------------- packet
class Packet:
    """Gaussian packet scattering off the barrier, integrated over k."""

    def __init__(self, E_mean=7.5, E_var=2.0, x0=-10.0, V0=8.0, a=2.0,
                 n_k=6000, sigmas=8.0, x_range=None, n_x=800):
        self.E_mean, self.E_var, self.x0 = E_mean, E_var, x0
        self.V0, self.a = V0, a
        self.b, self.l = packet_parameters(E_mean, E_var)
        self.sigma_k = np.sqrt(self.b)          # std of |phi(k)|^2

        lo = max(1e-6, self.l - sigmas * self.sigma_k)
        self.ks = np.linspace(lo, self.l + sigmas * self.sigma_k, n_k)
        self.dk = self.ks[1] - self.ks[0]
        self.phi = phi_k(self.ks, self.b, self.l, x0)

        if x_range is None:
            x_range = (2.5 * x0, -2.5 * x0)
        self.xs = np.linspace(*x_range, n_x)
        self.dx = self.xs[1] - self.xs[0]
        self.psi = psi_matrix(self.xs, self.ks, V0, a)
        self.T = transmission(self.ks, V0, a)

    def at(self, t):
        """Psi(x, t) on the position grid."""
        phase = np.exp(-1j * HBAR * self.ks ** 2 * t / (2 * MASS))
        return (self.psi @ (self.phi * phase)) * self.dk / np.sqrt(2 * np.pi)

    def density(self, t):
        return np.abs(self.at(t)) ** 2

    def spectral_transmission(self):
        """Predicted transmitted fraction: the T-weighted average over the spectrum."""
        w = np.abs(self.phi) ** 2
        return float(np.sum(w * self.T) / np.sum(w))

    def split(self, t):
        """(reflected, inside, transmitted) probability at time t, by integration."""
        rho = self.density(t)
        left = self.xs < 0
        mid = (self.xs >= 0) & (self.xs <= self.a)
        right = self.xs > self.a
        tot = np.trapezoid(rho, self.xs)
        return (float(np.trapezoid(rho[left], self.xs[left]) / tot),
                float(np.trapezoid(rho[mid], self.xs[mid]) / tot),
                float(np.trapezoid(rho[right], self.xs[right]) / tot))


# ------------------------------------------------------------- tunneling time
def phase_time(k, V0, a, dk=1e-5):
    """
    Wigner phase time for transmission: tau = hbar d(arg t)/dE, by central difference,
    where t = F e^{ika} is the transmission amplitude referred to the far edge of the
    barrier (so that psi_III = t e^{ik(x-a)} and tau is the delay of the transmitted
    peak relative to a free particle covering the same distance).

    For E < V0 this saturates at 2 / (v kappa) as the barrier widens — the Hartman
    effect — while the classical traversal time a/v keeps growing.
    """
    ks = np.array([k - dk, k + dk])
    _, _, _, F = coefficients(ks, V0, a)
    amp = F * np.exp(1j * ks * a)
    dphi = np.angle(amp[1] / amp[0])          # two-point phase difference, unwrapped
    dE = (HBAR ** 2 / (2 * MASS)) * (ks[1] ** 2 - ks[0] ** 2)
    return float(HBAR * dphi / dE)


def hartman_asymptote(k, V0):
    """Limiting phase time for a thick barrier, 2 / (v kappa)."""
    kappa = np.sqrt(2 * MASS * (V0 - (HBAR * k) ** 2 / (2 * MASS))) / HBAR
    v = HBAR * k / MASS
    return float(2.0 / (v * kappa))
