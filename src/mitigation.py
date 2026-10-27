"""Error mitigation: ZNE, CDR and M0 (M0 comes later)."""

from itertools import combinations
from math import comb

import numpy as np
from qiskit import transpile
from mitiq.zne.inference import ExpFactory, LinearFactory, PolyFactory
from mitiq.zne.scaling import fold_gates_at_random, fold_global
from qiskit.circuit.library import RXGate, RYGate, RZGate

from .backend import get_backend, get_noisy_estimator, get_ideal_estimator, get_transpiled
from .utils import get_config


class ZNE:
    """Zero-noise extrapolation with Mitiq.

    run() collects the noisy energies once; extrapolate() then fits them with
    linear / poly / exp, so we can compare fits on the plot without rerunning.
    """

    FOLDING = {"random": fold_gates_at_random, "global": fold_global}

    def __init__(self):
        cfg = get_config()["mitigation"]["zne"]
        self.noise_factors = [float(s) for s in cfg["noise_factors"]]
        self.fold = self.FOLDING[cfg.get("folding", "global")]
        self.poly_order = cfg.get("poly_order", 2)
        self.seed = get_config()["vqe"]["seed"]
        self.lambdas = None
        self.energies = None
        self.asymptote = None

    @staticmethod
    def maximally_mixed_asymptote(hamiltonian):
        """Tr(H)/2^n = coefficient of the all-identity Pauli term (others are traceless)."""
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if not (pauli.x.any() or pauli.z.any()):
                return coeff.real
        return 0.0

    def run(self, transpiled_circuit, transpiled_hamiltonian, e_nuc):
        """Noisy total energies at each noise scale factor."""
        estimator = get_noisy_estimator()

        def executor(circuit):
            return float(estimator.run([(circuit, transpiled_hamiltonian)]).result()[0].data.evs)

        energies = []
        for s in self.noise_factors:
            if self.fold is fold_gates_at_random:
                scaled = self.fold(transpiled_circuit, s, seed=self.seed)
            else:
                scaled = self.fold(transpiled_circuit, s)
            # Folding adds gates like sxdg that aren't native to the backend, and the
            # noise model gives non-native gates no error. Re-translate them to native
            # gates; optimization_level=0 so U^dagger U is not cancelled.
            scaled = transpile(scaled, get_backend(), optimization_level=0)
            energies.append(executor(scaled) + e_nuc)

        self.lambdas = np.array(self.noise_factors)
        self.energies = np.array(energies)
        self.asymptote = self.maximally_mixed_asymptote(transpiled_hamiltonian) + e_nuc
        return self.lambdas, self.energies

    def extrapolate(self, method="linear"):
        """Zero-noise energy and the fitted curve (for plotting) for 'linear', 'poly' or 'exp'."""
        if method == "linear":
            out = LinearFactory.extrapolate(self.lambdas, self.energies, full_output=True)
        elif method == "poly":
            out = PolyFactory.extrapolate(self.lambdas, self.energies, self.poly_order, full_output=True)
        elif method == "exp":
            out = ExpFactory.extrapolate(
                self.lambdas, self.energies, asymptote=self.asymptote, full_output=True
            )
        else:
            raise ValueError(f"method must be 'linear', 'poly' or 'exp', got '{method}'")
        zne_value, curve = out[0], out[-1]
        return zne_value, curve

    def save(self, path):
        np.savez(path, lambdas=self.lambdas, energies=self.energies, asymptote=self.asymptote)

    def load(self, path):
        data = np.load(path)
        self.lambdas = data["lambdas"]
        self.energies = data["energies"]
        self.asymptote = float(data["asymptote"])
        return self.lambdas, self.energies


class CDR:
    """Clifford data regression with C(n, k) training circuits and a linear fit."""

    ROTATIONS = {"rx": RXGate, "ry": RYGate, "rz": RZGate}

    def __init__(self):
        cfg = get_config()["mitigation"]["cdr"]
        self.k = cfg["k"]
        self.tol = cfg.get("clifford_tol", 1e-6)
        self.max_circuits = cfg.get("max_circuits")
        self.seed = cfg.get("seed")
        self.X_noisy = None
        self.X_exact = None
        self.a1 = None
        self.a2 = None

    # --- your notebook functions

    @staticmethod
    def nearest_clifford_angle(a):
        return (np.pi / 2) * np.round(a / (np.pi / 2))

    def non_clifford_positions(self, qc):
        """Indices in qc.data of all rx/ry/rz gates with non-Clifford angles."""
        positions = []
        for i, inst in enumerate(qc.data):
            if inst.operation.name in self.ROTATIONS:
                theta = float(inst.operation.params[0])
                k = theta / (np.pi / 2)
                if abs(k - round(k)) > self.tol:
                    positions.append(i)
        return positions

    def training_circuit(self, qc, positions, keep):
        """Replace every non-Clifford gate except the ones whose index j is in 'keep'."""
        circuit = qc.copy()
        keep = set(keep)
        for j, index in enumerate(positions):
            if j in keep:
                continue
            inst = circuit.data[index]
            gate_cls = self.ROTATIONS[inst.operation.name]
            new_angle = self.nearest_clifford_angle(float(inst.operation.params[0]))
            circuit.data[index] = inst.replace(operation=gate_cls(new_angle))
        return circuit

    # --- pipeline

    @staticmethod
    def to_rotation_gates(qc):
        """Rewrite a bound logical circuit with rx/ry/rz + Clifford gates (for CDR before transpiling)."""
        return transpile(
            qc, basis_gates=["rx", "ry", "rz", "cx", "h", "s", "sdg", "x", "sx"], optimization_level=0
        )

    def training_circuits(self, qc):
        """All C(n, k) training circuits, or a random sample of max_circuits if there are more.

        qc must be a bound circuit made of rx/ry/rz (+ Clifford) gates:
          before transpiling -> CDR.to_rotation_gates(bound_ansatz)
          after transpiling  -> the transpiled circuit
        """
        positions = self.non_clifford_positions(qc)
        n = len(positions)
        total = comb(n, self.k)
        print(f"n = {n} non-Clifford gates, k = {self.k} -> C(n, k) = {total}")

        if self.max_circuits is None or total <= self.max_circuits:
            keeps = list(combinations(range(n), self.k))
        else:
            rng = np.random.default_rng(self.seed)
            keeps = set()
            while len(keeps) < self.max_circuits:
                keeps.add(tuple(sorted(rng.choice(n, size=self.k, replace=False))))
            keeps = sorted(keeps)
            print(f"Sampled {self.max_circuits} of {total} circuits")

        return [self.training_circuit(qc, positions, keep) for keep in keeps]

    def training_data(self, circuits, hamiltonian, e_nuc, transpiled=True):
        """Noisy and exact energies of the training circuits.

        transpiled=False -> circuits are transpiled here (replacement BEFORE transpiling)
        transpiled=True  -> circuits are already transpiled (replacement AFTER transpiling)
        """
        pubs = []
        for circ in circuits:
            if transpiled:
                pubs.append((circ, hamiltonian.apply_layout(circ.layout)))
            else:
                pubs.append(get_transpiled(circ, [], hamiltonian))

        exact = get_ideal_estimator().run(pubs).result()
        noisy = get_noisy_estimator().run(pubs).result()
        self.X_exact = np.array([r.data.evs for r in exact]) + e_nuc
        self.X_noisy = np.array([r.data.evs for r in noisy]) + e_nuc
        return self.X_noisy, self.X_exact

    def fit(self):
        """Linear fit E_exact = a1 * E_noisy + a2."""
        self.a1, self.a2 = np.polyfit(self.X_noisy, self.X_exact, deg=1)
        return self.a1, self.a2

    def mitigate(self, noisy_energy):
        """Apply the fit to a noisy energy (or an array of them)."""
        return self.a1 * np.asarray(noisy_energy) + self.a2

    def save(self, path):
        np.savez(path, X_noisy=self.X_noisy, X_exact=self.X_exact)

    def load(self, path):
        data = np.load(path)
        self.X_noisy, self.X_exact = data["X_noisy"], data["X_exact"]
        return self.X_noisy, self.X_exact


class M0:
    """M0 mitigation via SlowQuant (to be implemented)."""
    pass