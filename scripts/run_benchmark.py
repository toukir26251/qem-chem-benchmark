"""Benchmark: EfficientSU2, UCCSD and fUCCSD with ZNE and CDR (same steps as 01_H2_BM.ipynb).

Usage (from the repo root):
    nohup python scripts/run_benchmark.py configs/h2.yaml &
    nohup python scripts/run_benchmark.py configs/lih.yaml &

Text output -> outputs/<molecule>_<date>.log
Plots       -> plots/
Raw data    -> data/
"""

import os
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
from src.backend import run_vqe, get_transpiled
from src.mitigation import ZNE, CDR

# ---------------------------------------------------------------- Setup

cfg = load_config(sys.argv[1])
MOL = cfg["molecule"]["name"]
date = datetime.now().strftime("%Y-%m-%d")

for d in ("data", "plots", "outputs"):
    os.makedirs(d, exist_ok=True)

# Everything printed goes to outputs/<molecule>_<date>.log
sys.stdout = open(f"outputs/{MOL}_{date}.log", "w", buffering=1)
sys.stderr = sys.stdout

molecule, mf, cas = get_molecule()
e_nuc = molecule.energy_nuc()

E_hf = mf.e_tot
E_fci = mcscf.CASCI(mf, cas[1], cas[0]).run().e_tot
print(f"HF  : {E_hf:.6f} Ha")
print(f"FCI : {E_fci:.6f} Ha")

ANSATZE = ["EfficientSU2", "UCCSD", "fUCCSD"]
results = {name: {} for name in ANSATZE}

# ---------------------------------------------------------------- VQE (noiseless) and transpiling

print("\n## VQE")
for name in ANSATZE:
    ansatz, hamiltonian, x0 = get_ansatz(name)
    vqe_energy, theta_opt = run_vqe(ansatz, hamiltonian, x0)
    transpiled_ansatz, transpiled_hamiltonian = get_transpiled(ansatz, theta_opt, hamiltonian)

    results[name].update(
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        theta_opt=theta_opt,
        transpiled_ansatz=transpiled_ansatz,
        transpiled_hamiltonian=transpiled_hamiltonian,
        E_exact=vqe_energy + e_nuc,
    )
    print(f"{name:13s} params = {ansatz.num_parameters:3d}   "
          f"E_VQE = {vqe_energy + e_nuc:.6f} Ha   "
          f"error vs FCI = {abs(vqe_energy + e_nuc - E_fci):.2e} Ha   "
          f"transpiled depth = {transpiled_ansatz.depth()}")
    print(theta_opt)

# ---------------------------------------------------------------- ZNE

print("\n## ZNE")
for name in ANSATZE:
    r = results[name]
    zne = ZNE()
    zne.run(r["transpiled_ansatz"], r["transpiled_hamiltonian"], e_nuc)
    zne.save(f"data/{MOL}_{name}_ZNE_{date}.npz")
    r["zne"] = zne
    r["E_noisy"] = zne.energies[0]
    print(f"{name:13s} energies at λ = {zne.lambdas}: {np.round(zne.energies, 6)}")

fig, axes = plt.subplots(1, len(ANSATZE), figsize=(5 * len(ANSATZE), 4), sharey=True)
x = np.linspace(0, max(ZNE().noise_factors), 100)

for ax, name in zip(axes, ANSATZE):
    r = results[name]
    zne = r["zne"]
    ax.plot(zne.lambdas, zne.energies, "ko", label="Noisy evaluations")
    for method in ["linear", "poly", "exp"]:
        E0, curve = zne.extrapolate(method)
        ax.plot(x, [curve(xi) for xi in x], "--", label=f"{method}: {E0:.5f}")
    ax.axhline(r["E_exact"], color="gray", ls=":", label="Exact")
    ax.set_title(name)
    ax.set_xlabel("Noise scale factor λ")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Total energy (Ha)")
plt.tight_layout()
plt.savefig(f"plots/{MOL}_ZNE_{date}.png", dpi=200)
plt.close()

# Extrapolation per ansatz, from the config (default: linear)
zne_method = cfg["mitigation"]["zne"].get("method", {})

for name in ANSATZE:
    r = results[name]
    method = zne_method.get(name, "linear")
    r["E_zne"], _ = r["zne"].extrapolate(method)
    print(f"{name:13s} ({method:6s})  Noisy error = {abs(r['E_noisy'] - r['E_exact']):.6f}   "
          f"ZNE error = {abs(r['E_zne'] - r['E_exact']):.6f} Ha")

# ---------------------------------------------------------------- CDR

print("\n## CDR")
for name in ANSATZE:
    r = results[name]
    cdr = CDR()
    print(f"{name}: ", end="")
    circuits = cdr.training_circuits(r["transpiled_ansatz"])
    cdr.training_data(circuits, r["hamiltonian"], e_nuc)
    cdr.save(f"data/{MOL}_{name}_CDR_{date}.npz")
    r["cdr"] = cdr

fig, axes = plt.subplots(1, len(ANSATZE), figsize=(5 * len(ANSATZE), 4))

for ax, name in zip(axes, ANSATZE):
    r = results[name]
    cdr = r["cdr"]
    a1, a2 = cdr.fit()
    r["E_cdr"] = float(cdr.mitigate(r["E_noisy"]))

    x = np.linspace(cdr.X_noisy.min(), cdr.X_noisy.max(), 100)
    ax.scatter(cdr.X_noisy, cdr.X_exact, s=8, label="Training circuits")
    ax.plot(x, cdr.mitigate(x), "r-", label=f"a1 = {a1:.3f}, a2 = {a2:.3f}")
    ax.set_title(name)
    ax.set_xlabel("Noisy energy (Ha)")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Exact energy (Ha)")
plt.tight_layout()
plt.savefig(f"plots/{MOL}_CDR_{date}.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- Summary

print("\n## Summary")
print(f"{'Ansatz':13s} {'Exact':>11s} {'Noisy':>11s} {'ZNE':>11s} {'CDR':>11s}   "
      f"{'err Noisy':>10s} {'err ZNE':>10s} {'err CDR':>10s}")
for name in ANSATZE:
    r = results[name]
    errs = [abs(r[k] - r["E_exact"]) for k in ("E_noisy", "E_zne", "E_cdr")]
    print(f"{name:13s} {r['E_exact']:11.6f} {r['E_noisy']:11.6f} {r['E_zne']:11.6f} {r['E_cdr']:11.6f}   "
          + " ".join(f"{e:10.2e}" for e in errs))

labels = ["Noisy", "ZNE", "CDR"]
keys = ["E_noisy", "E_zne", "E_cdr"]
width = 0.25
x = np.arange(len(ANSATZE))

plt.figure(figsize=(8, 5))
for i, (label, key) in enumerate(zip(labels, keys)):
    errors = [abs(results[n][key] - results[n]["E_exact"]) for n in ANSATZE]
    plt.bar(x + (i - 1) * width, errors, width, label=label)

plt.axhline(1.6e-3, color="gray", ls="--", label="Chemical accuracy")
plt.xticks(x, ANSATZE)
plt.yscale("log")
plt.ylabel("|E − E_exact| (Ha)")
plt.title(f"{MOL}: error before and after mitigation")
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{MOL}_mitigation_errors_{date}.png", dpi=200)
plt.close()

print("\nDone.")