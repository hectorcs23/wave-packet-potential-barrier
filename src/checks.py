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
    # --- 1. coefficients and stationary states
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4))
    E_grid = ks ** 2 / 2
    m = E_grid < 25
    ax[0].plot(E_grid[m], T[m], color="#2a6fdb", lw=2, label="T(E)")
    ax[0].plot(E_grid[m], R[m], color="#d9534f", lw=1.6, label="R(E)")
    ax[0].plot(E_grid[m], (T + R)[m], ":", color="#2e9e5b", lw=1.4, label="T + R")
    ax[0].axvline(V0, color="#333", ls="--", lw=1)
    ax[0].text(V0 + 0.3, 0.55, "$V_0$", fontsize=11)
    ax[0].set_xlabel("E"); ax[0].set_ylabel("probability")
    ax[0].set_ylim(-0.03, 1.08)
    ax[0].set_title("Below $V_0$ it tunnels; above it resonates",
                    loc="left", fontweight="bold", fontsize=12)
    ax[0].legend(fontsize=9)

    xs_s = np.linspace(-6, 10, 3000)
    for E, color, ls in ((4.0, "#d9534f", "-"), (12.0, "#2a6fdb", "-")):
        k = np.sqrt(2 * E)
        psi = bar.psi_matrix(xs_s, [k], V0, A)[:, 0]
        ax[1].plot(xs_s, np.abs(psi) ** 2, color=color, ls=ls, lw=1.6,
                   label=f"E = {E:.0f}   T = {bar.transmission([k], V0, A)[0]:.3g}")
    ax[1].axvspan(0, A, color="#9aa4b1", alpha=.25, lw=0)
    ax[1].text(A / 2, ax[1].get_ylim()[1] * 0.92, "barrier", ha="center", fontsize=9, color="#555")
    ax[1].set_xlabel("x"); ax[1].set_ylabel(r"$|\psi_k(x)|^2$")
    ax[1].set_title("Stationary states either side of the barrier",
                    loc="left", fontweight="bold", fontsize=12)
    ax[1].legend(fontsize=9)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGS / "coefficients.png", dpi=130)

    # --- 2. the packet splitting
    fig2, ax2 = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for t, color in zip((0, 2.6, 4, 8), ("#9aa4b1", "#e0a100", "#2e9e5b", "#2a6fdb")):
        ax2[0].plot(pk.xs, pk.density(t), color=color, lw=1.5, label=f"t = {t}")
    ax2[0].axvspan(0, A, color="#d9534f", alpha=.25, lw=0)
    ax2[0].set_xlim(-32, 32)
    ax2[0].set_xlabel("x"); ax2[0].set_ylabel(r"$|\Psi(x,t)|^2$")
    ax2[0].set_title(r"The packet splits at the barrier",
                     loc="left", fontweight="bold", fontsize=12)
    ax2[0].legend(fontsize=9)

    ts = np.linspace(0, 16, 60)
    splits = np.array([pk.split(t) for t in ts])
    ax2[1].plot(ts, splits[:, 0], color="#d9534f", lw=1.8, label="reflected (x < 0)")
    ax2[1].plot(ts, splits[:, 2], color="#2a6fdb", lw=1.8, label="transmitted (x > a)")
    ax2[1].plot(ts, splits[:, 1], color="#9aa4b1", lw=1.4, label="inside the barrier")
    ax2[1].axhline(T_spec, color="#2a6fdb", ls="--", lw=1)
    ax2[1].text(11.2, T_spec + 0.035, rf"$\langle T\rangle_\phi$ = {T_spec:.4f}",
                color="#2a6fdb", fontsize=9.5)
    ax2[1].axhline(report["T_at_mean_energy"], color="#333", ls=":", lw=1)
    ax2[1].text(0.4, report["T_at_mean_energy"] + 0.035,
                rf"$T(\langle E\rangle)$ = {report['T_at_mean_energy']:.4f}",
                color="#333", fontsize=9.5)
    ax2[1].set_xlabel("t"); ax2[1].set_ylabel("probability")
    ax2[1].set_ylim(-0.03, 1.05)
    ax2[1].set_title("Set by the spectrum, not the mean energy",
                     loc="left", fontweight="bold", fontsize=12)
    ax2[1].legend(fontsize=9, loc="center right")
    for a_ in ax2:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=.25)
    fig2.tight_layout()
    fig2.savefig(FIGS / "packet_split.png", dpi=130)

    # --- 3. Hartman
    fig3, ax3 = plt.subplots(figsize=(6.8, 4.5))
    ax3.plot(widths, taus, color="#2a6fdb", lw=2, label=r"phase time $\tau_\phi$")
    ax3.plot(widths, widths / k_tun, "--", color="#d9534f", lw=1.6,
             label=r"classical $a/v$")
    ax3.axhline(asym, color="#2e9e5b", ls=":", lw=1.4,
                label=rf"$2/(v\kappa)$ = {asym:.3f}")
    ax3.set_xlabel("barrier width a")
    ax3.set_ylabel("time")
    ax3.set_title(f"Hartman effect at E = {E_tun:.0f} < $V_0$ = {V0:.0f}",
                  loc="left", fontweight="bold", fontsize=12)
    ax3.legend(fontsize=9, loc="upper left")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(alpha=.25)
    ax3b = ax3.twinx()
    ax3b.semilogy(widths, Ts, color="#9aa4b1", lw=1.3)
    ax3b.set_ylabel("T (grey, log scale)", color="#777")
    ax3b.tick_params(axis="y", colors="#777")
    fig3.tight_layout()
    fig3.savefig(FIGS / "hartman.png", dpi=130)

    np.savetxt(RESULTS / "hartman_phase_time.csv",
               np.column_stack([widths, taus, widths / k_tun, Ts]),
               delimiter=",", header="width,phase_time,classical_time,T", comments="")
    np.savetxt(RESULTS / "packet_split.csv",
               np.column_stack([ts, splits]), delimiter=",",
               header="t,reflected,inside,transmitted", comments="")

    print(f"\nwrote {RESULTS}/ and {FIGS}/coefficients.png, packet_split.png, hartman.png")


if __name__ == "__main__":
    main()
