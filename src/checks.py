"""
checks.py
=========

Numerical verification of everything the report claims, plus the figures used in
the README. Nothing here is read off a plot: each number is recomputed and
compared against an independent route.

    1. Unitarity            T(E) + R(E) = 1 for every E
    2. Algebra              the closed-form T equals |F|^2 from the coefficients
    3. Boundary conditions  psi and psi' continuous at x = 0 and x = a
    4. Packet parameters    the analytic b, l reproduce the requested <E>, var(E)
    5. Norm                 |Psi(x,t)|^2 integrates to 1 at every time
    6. The split            the transmitted fraction of the propagated packet
                            equals the T-weighted average over its spectrum
    7. Hartman              the phase time saturates with barrier width while the
                            classical traversal time does not

Usage:
    python src/checks.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import barrier as bar

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "img"
RESULTS = ROOT / "results"

V0, A = 8.0, 2.0
E_MEAN, E_VAR, X0 = 7.5, 2.0, -10.0


def banner(n, text):
    print(f"\n[{n}] {text}\n" + "-" * (len(text) + 5))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    report = {}

    # ------------------------------------------------ 1 & 2: T, R, unitarity
    banner(1, "Unitarity and the closed-form coefficients")
    ks = np.linspace(0.05, 9.0, 20_000)
    T = bar.transmission(ks, V0, A)
    B, C, D, F = bar.coefficients(ks, V0, A)
    R = np.abs(B) ** 2
    err_unit = float(np.max(np.abs(T + R - 1)))
    err_alg = float(np.max(np.abs(T - np.abs(F) ** 2)))
    print(f"max |T + R - 1|       = {err_unit:.3e}")
    print(f"max |T - |F|^2|       = {err_alg:.3e}")
    report["max_unitarity_error"] = err_unit
    report["max_closed_form_error"] = err_alg

    # ------------------------------------------------ 3: boundary conditions
    banner(3, "Continuity of psi and psi' at the barrier edges")
    rows = []
    for E in (2.0, 4.0, 7.5, 12.0, 20.0):
        k = np.sqrt(2 * E)
        d0, dd0, da, dda = bar.continuity_residuals(k, V0, A)
        rows.append((d0, dd0, da, dda))
        print(f"E = {E:5.1f}   x = 0: |Δψ| = {d0:.2e}  |Δψ'| = {dd0:.2e}   "
              f"x = a: |Δψ| = {da:.2e}  |Δψ'| = {dda:.2e}")
    report["max_psi_jump"] = max(max(r[0], r[2]) for r in rows)
    report["max_dpsi_jump"] = max(max(r[1], r[3]) for r in rows)

    # ------------------------------------------------ 4: packet parameters
    banner(4, "Packet parameters from <E> and var(E)")
    pk = bar.Packet(E_mean=E_MEAN, E_var=E_VAR, x0=X0, V0=V0, a=A,
                    n_k=4000, n_x=2200, x_range=(-75, 75))
    w = np.abs(pk.phi) ** 2
    k_bar = float(np.sum(w * pk.ks) / np.sum(w))
    k_var = float(np.sum(w * (pk.ks - k_bar) ** 2) / np.sum(w))
    E_num = pk.ks ** 2 / 2
    E_bar = float(np.sum(w * E_num) / np.sum(w))
    E_var_num = float(np.sum(w * (E_num - E_bar) ** 2) / np.sum(w))
    print(f"analytic  b = {pk.b:.6f}   l = {pk.l:.6f}   sigma_k = {pk.sigma_k:.6f}")
    print(f"numeric   <k> = {k_bar:.6f}  var(k) = {k_var:.6f}")
    print(f"numeric   <E> = {E_bar:.6f} (asked {E_MEAN})   "
          f"var(E) = {E_var_num:.6f} (asked {E_VAR})")
    report["E_mean_numeric"] = E_bar
    report["E_var_numeric"] = E_var_num

    # ------------------------------------------------ 5 & 6: the split
    banner(6, "Propagating the packet and splitting the probability")
    T_spec = pk.spectral_transmission()
    print(f"spectral prediction  <T> = {T_spec:.6f}")
    print(f"T at the mean energy     = {bar.transmission([np.sqrt(2 * E_MEAN)], V0, A)[0]:.6f}")
    above = E_num > V0
    frac_above = float(np.sum(w[above]) / np.sum(w))
    share = float(np.sum(w[above] * pk.T[above]) / np.sum(w * pk.T))
    print(f"spectral weight above V0 = {frac_above:.4f}")
    print(f"share of the transmitted probability coming from E > V0 = {share:.4f}")

    times = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    table = []
    for t in times:
        norm = float(np.trapezoid(pk.density(t), pk.xs))
        r, m, tr = pk.split(t)
        table.append((t, norm, r, m, tr))
        print(f"t = {t:5.1f}   norm = {norm:.6f}   R = {r:.5f}   inside = {m:.2e}   T = {tr:.5f}")
    T_late = table[-1][4]
    print(f"\npropagated T = {T_late:.6f} vs spectral {T_spec:.6f}   "
          f"(relative difference {abs(T_late - T_spec) / T_spec:.2e})")
    report["T_spectral"] = T_spec
    report["T_propagated"] = T_late
    report["T_at_mean_energy"] = float(bar.transmission([np.sqrt(2 * E_MEAN)], V0, A)[0])
    report["norm_drift"] = max(abs(row[1] - 1) for row in table)
    report["weight_above_V0"] = frac_above
    report["transmission_share_above_V0"] = share

    # ------------------------------------------------ 7: Hartman
    banner(7, "Phase time vs barrier width (Hartman effect)")
    E_tun = 4.0
    k_tun = np.sqrt(2 * E_tun)
    widths = np.linspace(0.2, 10.0, 60)
    taus = np.array([bar.phase_time(k_tun, V0, a) for a in widths])
    Ts = np.array([bar.transmission([k_tun], V0, a)[0] for a in widths])
    asym = bar.hartman_asymptote(k_tun, V0)
    print(f"E = {E_tun} < V0 = {V0}")
    for a in (1, 2, 4, 6, 8, 10):
        tau = bar.phase_time(k_tun, V0, a)
        print(f"  a = {a:4.1f}   tau_phase = {tau:.5f}   classical a/v = {a / k_tun:.5f}   "
              f"T = {bar.transmission([k_tun], V0, a)[0]:.2e}")
    print(f"  asymptote 2/(v*kappa) = {asym:.5f}; "
          f"value at a = 10 is {taus[-1]:.5f} ({100 * abs(taus[-1] - asym) / asym:.2f} % off)")
    report["hartman_asymptote"] = asym
    report["tau_at_a10"] = float(taus[-1])

    with open(RESULTS / "checks.txt", "w") as fh:
        for k_, v_ in report.items():
            fh.write(f"{k_}\t{v_}\n")

    # ================================================================ figures
    # --- 1. transmission and reflection coefficients
    E_grid = ks ** 2 / 2
    m = E_grid < 25
    plt.figure()
    plt.plot(E_grid[m], T[m], label="T(E)")
    plt.plot(E_grid[m], R[m], label="R(E)")
    plt.plot(E_grid[m], (T + R)[m], label="T + R")
    plt.plot([V0, V0], [0, 1], "k--", label="$V_0$")
    plt.xlabel("E")
    plt.ylabel("Probability")
    plt.title(f"Transmission and reflection, $V_0$ = {V0:.0f}, a = {A:.0f}")
    plt.legend()
    plt.savefig(FIGS / "coefficients.png")

    # --- 2. stationary states either side of the barrier
    xs_s = np.linspace(-6, 10, 3000)
    plt.figure()
    top = 0.0
    for E in (4.0, 12.0):
        k = np.sqrt(2 * E)
        dens = np.abs(bar.psi_matrix(xs_s, [k], V0, A)[:, 0]) ** 2
        top = max(top, dens.max())
        plt.plot(xs_s, dens,
                 label=f"E = {E:.0f}, T = {bar.transmission([k], V0, A)[0]:.3g}")
    plt.plot([0, 0], [0, top], "k--", label="Barrier edges")
    plt.plot([A, A], [0, top], "k--")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi_k(x)|^2$")
    plt.title("Stationary states below and above the barrier")
    plt.legend()
    plt.savefig(FIGS / "stationary_states.png")

    # --- 3. the packet splitting
    plt.figure()
    top = 0.0
    for t in (0, 2.6, 4, 8):
        dens = pk.density(t)
        top = max(top, dens.max())
        plt.plot(pk.xs, dens, label=f"t = {t}")
    plt.plot([0, 0], [0, top], "k--", label="Barrier edges")
    plt.plot([A, A], [0, top], "k--")
    plt.xlim(-32, 32)
    plt.xlabel("x")
    plt.ylabel(r"$|\Psi(x,t)|^2$")
    plt.title("The packet splits at the barrier")
    plt.legend()
    plt.savefig(FIGS / "packet_split.png")

    # --- 4. reflected / transmitted fractions vs time
    ts = np.linspace(0, 16, 60)
    splits = np.array([pk.split(t) for t in ts])
    plt.figure()
    plt.plot(ts, splits[:, 0], label="Reflected (x < 0)")
    plt.plot(ts, splits[:, 2], label="Transmitted (x > a)")
    plt.plot(ts, splits[:, 1], label="Inside the barrier")
    plt.plot([ts[0], ts[-1]], [T_spec, T_spec], "k--",
             label=rf"$\langle T\rangle_\phi$ = {T_spec:.4f}")
    plt.plot([ts[0], ts[-1]], [report["T_at_mean_energy"]] * 2, "k:",
             label=rf"$T(\langle E\rangle)$ = {report['T_at_mean_energy']:.4f}")
    plt.xlabel("t")
    plt.ylabel("Probability")
    plt.title("Transmitted fraction is set by the spectrum")
    plt.legend()
    plt.savefig(FIGS / "packet_fractions.png")

    # --- 5. Hartman effect
    plt.figure()
    plt.plot(widths, taus, label=r"Phase time $\tau_\phi$")
    plt.plot(widths, widths / k_tun, label="Classical a/v")
    plt.plot([widths[0], widths[-1]], [asym, asym], "k--",
             label=rf"$2/(v\kappa)$ = {asym:.3f}")
    plt.xlabel("Barrier width a")
    plt.ylabel("Time")
    plt.title(f"Hartman effect at E = {E_tun:.0f} < $V_0$ = {V0:.0f}")
    plt.legend()
    plt.savefig(FIGS / "hartman.png")

    # --- 6. transmission vs barrier width
    plt.figure()
    plt.semilogy(widths, Ts)
    plt.xlabel("Barrier width a")
    plt.ylabel("Transmission T")
    plt.title(f"Tunnelling probability vs barrier width, E = {E_tun:.0f}")
    plt.savefig(FIGS / "hartman_transmission.png")

    np.savetxt(RESULTS / "hartman_phase_time.csv",
               np.column_stack([widths, taus, widths / k_tun, Ts]),
               delimiter=",", header="width,phase_time,classical_time,T", comments="")
    np.savetxt(RESULTS / "packet_split.csv",
               np.column_stack([ts, splits]), delimiter=",",
               header="t,reflected,inside,transmitted", comments="")

    print(f"\nwrote {RESULTS}/ and the six figures in {FIGS}/")


if __name__ == "__main__":
    main()
