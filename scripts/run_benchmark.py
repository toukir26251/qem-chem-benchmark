"""Benchmark: EfficientSU2, UCCSD, fUCCSD and pp-tUPS with ZNE, CDR and M0.

Runs the same steps as notebooks/01_H2_benchmark.ipynb and 02_LiH_benchmark.ipynb,
with the molecule taken from the config file.

Usage (from the repo root):
    nohup python scripts/run_benchmark.py configs/h2.yaml &
    nohup python scripts/run_benchmark.py configs/lih.yaml &

Text output -> outputs/latest/<molecule>/<molecule>_<date>.log (previous run -> outputs/backup/<molecule>/<timestamp>/)
                 (every notebook cell's output is followed by a blank line)
Plots       -> plots/latest/<molecule>/   (previous run -> plots/backup/<molecule>/<timestamp>/)
Raw data    -> data/latest/<molecule>/    (previous run -> data/backup/<molecule>/<timestamp>/)
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no screen on the HPC, only save figures
import matplotlib.pyplot as plt
import numpy as np
from pyscf import mcscf

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.chem import get_molecule, get_ansatz
from src.backend import run_vqe, get_transpiled, get_noisy_energy
from src.mitigation import ZNE, CDR, M0

# ---------------------------------------------------------------- Setup

cfg = load_config(sys.argv[1])
MOL = cfg["molecule"]["name"]
date = datetime.now().strftime("%Y-%m-%d")

# Output folders: <kind>/latest/<molecule>/ ; the previous run of this molecule is moved to
# <kind>/backup/<molecule>/<timestamp>/ before anything new is written.
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
for kind in ("data", "plots", "outputs"):
    latest = Path(kind) / "latest" / MOL
    if latest.exists() and any(latest.iterdir()):
        dest = Path(kind) / "backup" / MOL / stamp
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(latest), str(dest))
    latest.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data") / "latest" / MOL
PLOT_DIR = Path("plots") / "latest" / MOL
OUT_DIR = Path("outputs") / "latest" / MOL

# Everything printed goes to outputs/latest/<molecule>/<molecule>_<date>.log
sys.stdout = open(f"{OUT_DIR}/{MOL}_{date}.log", "w", buffering=1)
sys.stderr = sys.stdout


def end_cell():
    """Blank line below the output of each notebook cell."""
    print()


def show(path, **kw):
    """Notebook plt.show(): save the figure instead (no screen) and close it."""
    if path:
        plt.savefig(path, dpi=200, **kw)
    plt.close()


molecule, mf, cas = get_molecule()
e_nuc = molecule.energy_nuc()

E_hf = mf.e_tot
E_fci = mcscf.CASCI(mf, cas[1], cas[0]).run().e_tot
print(f"HF  : {E_hf:.6f} Ha")
print(f"FCI : {E_fci:.6f} Ha")
end_cell()

ANSATZE = ["EfficientSU2", "UCCSD", "fUCCSD", "pp-tUPS"]

# Which error-mitigation methods run: whatever is listed under `mitigation:` in the config
# (zne, cdr, m0). To skip one, remove or comment it out in the yaml.
RUN = [m for m in ("zne", "cdr", "m0") if m in (cfg.get("mitigation") or {})]
print(f"Mitigation methods to run: {', '.join(RUN) if RUN else 'none (noisy only)'}")
end_cell()
SEEDS = cfg["vqe"]["seeds"]
results = {name: {seed: {} for seed in SEEDS} for name in ANSATZE}

# ---------------------------------------------------------------- VQE (noiseless) and transpiling

for name in ANSATZE:
    for seed in SEEDS:
        ansatz, hamiltonian, x0 = get_ansatz(name, seed)
        vqe_energy, theta_opt = run_vqe(ansatz, hamiltonian, x0)
        transpiled_ansatz, transpiled_hamiltonian = get_transpiled(ansatz, theta_opt, hamiltonian, seed)

        results[name][seed].update(ansatz=ansatz, hamiltonian=hamiltonian, theta_opt=theta_opt,
            transpiled_ansatz=transpiled_ansatz, transpiled_hamiltonian=transpiled_hamiltonian,
            E_exact=vqe_energy + e_nuc)
        print(f"{name:13s} seed={seed}  E_VQE = {vqe_energy + e_nuc:.6f}  "
              f"error vs FCI = {abs(vqe_energy + e_nuc - E_fci):.2e}  depth = {transpiled_ansatz.depth()}")

        print(f"seed : {seed}", len(results[name][seed]["theta_opt"]))
end_cell()

# ---------------------------------------------------------------- Noisy energy

# One noisy energy per (ansatz, seed); ZNE (lambda = 1), CDR and the M0 comparison all use this value.
for name in ANSATZE:
    for seed in SEEDS:
        r = results[name][seed]
        r["E_noisy"] = get_noisy_energy(r["transpiled_ansatz"], r["transpiled_hamiltonian"], e_nuc, seed)
        print(f"{name:13s} seed={seed}  E_noisy = {r['E_noisy']:.6f}  "
              f"noisy error = {abs(r['E_noisy'] - r['E_exact']):.6f}")
end_cell()

# ---------------------------------------------------------------- ZNE
if "zne" in RUN:

    for name in ANSATZE:
        for seed in SEEDS:
            r = results[name][seed]
            zne = ZNE(seed)
            zne.run(r["transpiled_ansatz"], r["transpiled_hamiltonian"], e_nuc, E_noisy=r["E_noisy"])
            zne.save(f"{DATA_DIR}/{MOL}_{name}_ZNE_seed{seed}_{date}.npz")
            r["zne"] = zne

    # All ansatze side by side
    METHODS = ["linear", "poly", "exp"]
    COLORS = {"linear": "tab:blue", "poly": "tab:orange", "exp": "tab:green"}
    x = np.linspace(0, max(ZNE(SEEDS[0]).noise_factors), 100)

    fig, axes = plt.subplots(1, len(ANSATZE), figsize=(5 * len(ANSATZE), 4), sharey=True)
    for ax, name in zip(axes, ANSATZE):
        for seed in SEEDS:
            zne = results[name][seed]["zne"]
            ax.plot(zne.lambdas, zne.energies, "ko", ms=3, alpha=0.6)
            for m in METHODS:
                _, curve = zne.extrapolate(m)
                ax.plot(x, [curve(xi) for xi in x], "--", color=COLORS[m], alpha=0.4, lw=1)
        ax.axhline(np.mean([results[name][s]["E_exact"] for s in SEEDS]), color="gray", ls=":", label="Exact")
        for m in METHODS:
            ax.plot([], [], "--", color=COLORS[m], label=m)
        ax.set_title(name); ax.set_xlabel("Noise scale factor λ"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Total energy (Ha)")
    plt.tight_layout(); show(f"{PLOT_DIR}/{MOL}_ZNE_{date}.png")

    # One figure per ansatz
    STYLES = {"linear": "--", "poly": "-.", "exp": ":"}
    x = np.linspace(0, max(ZNE(SEEDS[0]).noise_factors), 100)

    for name in ANSATZE:
        fig, ax = plt.subplots(figsize=(7, 5))

        for i, seed in enumerate(SEEDS):
            zne = results[name][seed]["zne"]
            c = f"C{i}"
            ax.plot(zne.lambdas, zne.energies, "o", color=c, label=f"seed {seed}")
            for m in METHODS:
                _, curve = zne.extrapolate(m)
                ax.plot(x, [curve(xi) for xi in x], STYLES[m], color=c, lw=1.2)
            ax.axhline(results[name][seed]["E_exact"], color=c, lw=0.8, alpha=0.5)
        ax.plot([], [], "-", color="k", lw=0.8, alpha=0.5, label="Exact")

        ax.set_title(f"{name}: ZNE")
        ax.set_xlabel("Noise scale factor λ")
        ax.set_ylabel("Total energy (Ha)")
        for m in METHODS:
            ax.plot([], [], STYLES[m], color="k", label=m)

        ax.legend(fontsize=8)
        plt.tight_layout()
        show(f"{PLOT_DIR}/{MOL}_ZNE_{name}_{date}.png")

    # Extrapolation per (ansatz, seed). If the config sets `zne: method: <ansatz>: linear|poly|exp`,
    # that fit is used; otherwise the fit with R^2 closest to 1.
    cfg_method = cfg["mitigation"]["zne"].get("method") or {}
    for name, m in cfg_method.items():
        if m is not None and m not in METHODS:
            raise ValueError(f"zne.method.{name} = {m!r}; use one of {METHODS} or leave it empty")
    zne_method = {name: {} for name in ANSATZE}       # zne_method[name][seed], filled at run time

    for name in ANSATZE:
        for seed in SEEDS:
            r = results[name][seed]
            err = {m: abs(r["zne"].extrapolate(m)[0] - r["E_exact"]) for m in METHODS}
            err = {m: (np.inf if np.isnan(e) else e) for m, e in err.items()}    # a NaN fit is never picked
            method_by_min_error = min(METHODS, key=err.get)

            r2 = {m: r["zne"].r_squared(m) for m in METHODS}
            method_by_r2 = min(METHODS, key=lambda m: np.inf if np.isnan(r2[m]) else abs(1 - r2[m]))

            fixed = cfg_method.get(name)
            if fixed:
                method = fixed
                print(f"Method set in config : '{method}' (R^2 = {r2[method]}); by R^2 would be '{method_by_r2}', "
                      f"by min error '{method_by_min_error}'")
            else:
                method = method_by_r2
                print(f"Method choosen : '{method} : {r2[method]}' by R^2, Got : '{method_by_min_error}' by min error")

            zne_method[name][seed] = method
            r["E_zne"], _ = r["zne"].extrapolate(method)
            print(f"{name:13s} ({method:6s}) Seed = {seed}")
            print(f"   Exact = {r['E_exact']:.6f}   Noisy = {r['E_noisy']:.6f}   ZNE = {r['E_zne']:.6f} Ha")
            print(f"   Noisy error = {abs(r['E_noisy'] - r['E_exact']):.6f}   "
                f"ZNE error = {abs(r['E_zne'] - r['E_exact']):.6f}   "
                f"err = " + " ".join(f"{m}:{err[m]:.2e}" for m in METHODS) + "\n")
    end_cell()

    print(f"{'Ansatz':13s} {'Noisy error (mean ± std)':>28s} {'ZNE error (mean ± std)':>28s}")
    for name in ANSATZE:
        noisy = np.array([abs(results[name][s]["E_noisy"] - results[name][s]["E_exact"]) for s in SEEDS])
        zne = np.array([abs(results[name][s]["E_zne"] - results[name][s]["E_exact"]) for s in SEEDS])
        print(f"{name:13s} {noisy.mean():14.6f} ± {noisy.std():10.6f} {zne.mean():14.6f} ± {zne.std():10.6f}")
    end_cell()

    w = 0.35
    x = np.arange(len(ANSATZE))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label, color) in enumerate([("E_noisy", "Noisy error", "tab:blue"), ("E_zne", "ZNE error", "tab:orange")]):
        errs_ = [np.array([abs(results[n][s][key] - results[n][s]["E_exact"]) for s in SEEDS]) for n in ANSATZE]
        mean = np.array([e.mean() for e in errs_])
        std = np.array([e.std() for e in errs_])
        bars = ax.bar(x + (i - 0.5) * w, mean, w, color=color, label=label)
        ax.bar_label(bars, labels=[f"{m:.4f} ± {sd:.4f}" for m, sd in zip(mean, std)], fontsize=6, rotation=90, padding=4)

    ax.axhline(1.6e-3, color="gray", ls="--", label="Chemical accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(ANSATZE)
    ax.set_ylabel(r"$|E - E_\mathrm{exact}|$ (Ha)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 50)   # headroom for the rotated labels
    ax.set_title(f"{MOL}: Noisy vs. ZNE error (bar labels: mean ± std over {len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    show(f"{PLOT_DIR}/{MOL}_ZNE_errors_{date}.png")


# ---------------------------------------------------------------- CDR
if "cdr" in RUN:

    for name in ANSATZE:
        for seed in SEEDS:
            print(f"{name}\n")
            r = results[name][seed]
            cdr = CDR(seed)
            circuits = cdr.training_circuits(r["transpiled_ansatz"])
            cdr.training_data(circuits, r["hamiltonian"], e_nuc)
            cdr.save(f"{DATA_DIR}/{MOL}_{name}_CDR_seed{seed}_{date}.npz")
            r["cdr"] = cdr
    end_cell()

    for name in ANSATZE:
        fig, ax = plt.subplots(figsize=(7, 5))

        for i, seed in enumerate(SEEDS):
            r = results[name][seed]; cdr = r["cdr"]
            a1, a2 = cdr.fit()
            r["E_cdr"] = float(cdr.mitigate(r["E_noisy"]))
            x = np.linspace(cdr.X_noisy.min(), cdr.X_noisy.max(), 100)
            ax.scatter(cdr.X_noisy, cdr.X_exact, s=8, color=f"C{i}", alpha=0.5)
            ax.plot(x, cdr.mitigate(x), "-", color=f"C{i}", label=f"seed {seed}: a1={a1:.3f}, a2={a2:.3f}")

        ax.set_title(f"{name}: CDR")
        ax.set_xlabel("Noisy energy (Ha)")
        ax.set_ylabel("Exact energy (Ha)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        show(f"{PLOT_DIR}/{MOL}_CDR_{name}_{date}.png")

    for name in ANSATZE:
        for seed in SEEDS:
            r = results[name][seed]
            print(f"{name:13s} (CDR) Seed = {seed}")
            print(f"   Exact = {r['E_exact']:.6f}   Noisy = {r['E_noisy']:.6f}   CDR = {r['E_cdr']:.6f} Ha")
            print(f"   Noisy error = {abs(r['E_noisy'] - r['E_exact']):.6f}   "
                  f"CDR error = {abs(r['E_cdr'] - r['E_exact']):.6f}")
    end_cell()

    print(f"{'Ansatz':13s} {'Noisy error (mean ± std)':>28s} {'CDR error (mean ± std)':>28s}")
    for name in ANSATZE:
        noisy = np.array([abs(results[name][s]["E_noisy"] - results[name][s]["E_exact"]) for s in SEEDS])
        cdr = np.array([abs(results[name][s]["E_cdr"] - results[name][s]["E_exact"]) for s in SEEDS])
        print(f"{name:13s} {noisy.mean():14.6f} ± {noisy.std():10.6f} {cdr.mean():14.6f} ± {cdr.std():10.6f}")
    end_cell()

    w = 0.35
    x = np.arange(len(ANSATZE))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label, color) in enumerate([("E_noisy", "Noisy error", "tab:blue"), ("E_cdr", "CDR error", "tab:green")]):
        errs_ = [np.array([abs(results[n][s][key] - results[n][s]["E_exact"]) for s in SEEDS]) for n in ANSATZE]
        mean = np.array([e.mean() for e in errs_])
        std = np.array([e.std() for e in errs_])
        bars = ax.bar(x + (i - 0.5) * w, mean, w, color=color, label=label)
        ax.bar_label(bars, labels=[f"{m:.4f} ± {sd:.4f}" for m, sd in zip(mean, std)], fontsize=6, rotation=90, padding=4)

    ax.axhline(1.6e-3, color="gray", ls="--", label="Chemical accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(ANSATZE)
    ax.set_ylabel(r"$|E - E_\mathrm{exact}|$ (Ha)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 50)   # headroom for the rotated labels
    ax.set_title(f"{MOL}: Noisy vs. CDR error (bar labels: mean ± std over {len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    show(f"{PLOT_DIR}/{MOL}_CDR_errors_{date}.png")


# ---------------------------------------------------------------- M0
if "m0" in RUN:

    M0_ANSATZE = ["fUCCSD", "pp-tUPS"]   # M0 only works with SlowQuant ansatze

    for name in ANSATZE:
        for seed in SEEDS:
            r = results[name][seed]
            if name in M0_ANSATZE:
                r["E_noisy_m0"], r["E_m0"] = M0(seed).run(name, r["theta_opt"])
                print(f"{name:13s} Seed = {seed}  Noisy error = {abs(r['E_noisy'] - r['E_exact']):.6f}   "
                      f"M0 error = {abs(r['E_m0'] - r['E_exact']):.6f} Ha")
            else:
                r["E_noisy_m0"] = r["E_m0"] = np.nan
    end_cell()

    print(f"{'Ansatz':13s} {'Noisy error (mean ± std)':>28s} {'M0 error (mean ± std)':>28s}")
    for name in M0_ANSATZE:
        noisy = np.array([abs(results[name][s]["E_noisy"] - results[name][s]["E_exact"]) for s in SEEDS])
        m0 = np.array([abs(results[name][s]["E_m0"] - results[name][s]["E_exact"]) for s in SEEDS])
        print(f"{name:13s} {noisy.mean():14.6f} ± {noisy.std():10.6f} {m0.mean():14.6f} ± {m0.std():10.6f}")
    end_cell()

    w = 0.35
    x = np.arange(len(M0_ANSATZE))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (key, label, color) in enumerate([("E_noisy", "Noisy error", "tab:blue"), ("E_m0", "M0 error", "tab:red")]):
        errs_ = [np.array([abs(results[n][s][key] - results[n][s]["E_exact"]) for s in SEEDS]) for n in M0_ANSATZE]
        mean = np.array([e.mean() for e in errs_])
        std = np.array([e.std() for e in errs_])
        bars = ax.bar(x + (i - 0.5) * w, mean, w, color=color, label=label)
        ax.bar_label(bars, labels=[f"{m:.4f} ± {sd:.4f}" for m, sd in zip(mean, std)], fontsize=6, rotation=90, padding=4)

    ax.axhline(1.6e-3, color="gray", ls="--", label="Chemical accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(M0_ANSATZE)
    ax.set_ylabel(r"$|E - E_\mathrm{exact}|$ (Ha)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 50)   # headroom for the rotated labels
    ax.set_title(f"{MOL}: Noisy vs. M0 error (bar labels: mean ± std over {len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    show(f"{PLOT_DIR}/{MOL}_M0_errors_{date}.png")

    for name in M0_ANSATZE:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(SEEDS))
        noisy = [results[name][s]["E_noisy"] for s in SEEDS]
        m0 = [results[name][s]["E_m0"] for s in SEEDS]
        exact = [results[name][s]["E_exact"] for s in SEEDS]

        ax.plot(x, noisy, "o", color="C0", label="Noisy")
        ax.plot(x, m0, "s", color="C3", label="M0")
        ax.plot(x, exact, "_", color="k", ms=25, mew=1.5, label="Exact")
        for xi, n, m in zip(x, noisy, m0):
            ax.annotate("", xy=(xi, m), xytext=(xi, n),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xticks(x, [f"seed {s}" for s in SEEDS])
        ax.set_title(f"{name}: M0")
        ax.set_ylabel("Total energy (Ha)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        show(f"{PLOT_DIR}/{MOL}_M0_{name}_{date}.png")


# ---------------------------------------------------------------- Summary

def errs(name, key):
    return np.array([abs(results[name][s][key] - results[name][s]["E_exact"]) for s in SEEDS])

# Noisy always; the others only if they were run
methods = [("E_noisy", "Noisy")] + [(k, l) for m, k, l in
           (("zne", "E_zne", "ZNE"), ("cdr", "E_cdr", "CDR"), ("m0", "E_m0", "M0")) if m in RUN]

print(f"{'Ansatz':13s} " + " ".join(f"{'err ' + label:>22s}" for _, label in methods))
for name in ANSATZE:
    cells = []
    for k, _ in methods:
        e = errs(name, k)
        cells.append(f"{'-':>22s}" if np.isnan(e).all() else f"{e.mean():10.2e} ± {e.std():8.2e}")
    print(f"{name:13s} " + " ".join(cells))
end_cell()

n_m = len(methods)
w = 0.8 / n_m          # 0.2 for four methods
x = np.arange(len(ANSATZE))

fig, ax = plt.subplots(figsize=(10, 5))
for i, (key, label) in enumerate(methods):
    mean = np.array([errs(n, key).mean() for n in ANSATZE])
    std = np.array([errs(n, key).std() for n in ANSATZE])
    bars = ax.bar(x + (i - (n_m - 1) / 2) * w, mean, w, label=label)
    ax.bar_label(bars, labels=[f"{m:.6f} ± {sd:.4f}" for m, sd in zip(mean, std)], fontsize=6, rotation=90, padding=4)

plt.axhline(1.6e-3, color="gray", ls="--", label="Chemical accuracy")

ax.set_xticks(x)
ax.set_xticklabels(ANSATZE)
ax.set_ylabel(r"$|E - E_\mathrm{exact}|$ (Ha)")
ax.set_yscale("log")
ax.set_ylim(top=ax.get_ylim()[1] * 50)   # headroom for the rotated labels
ax.set_title(f"{MOL}: Error vs. mitigation method (bar labels: mean ± std over {len(SEEDS)} seeds)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
show(f"{PLOT_DIR}/{MOL}_mitigation_errors_{date}.png", bbox_inches="tight")

print("Done.")
