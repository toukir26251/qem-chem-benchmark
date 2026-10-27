"""Molecules from the config, for both the SlowQuant and the Qiskit route."""

import contextlib
import io

import numpy as np
import pyscf
from pyscf import scf
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface

from .utils import get_config

def get_molecule(cfg=None):
    """SlowQuant route (fUCCSD, pp-tUPS). PySCF molecule, RHF and active space."""
    m = (cfg or get_config())['molecule']
    mol = pyscf.M(
        atom=m["geometry"],
        basis=m["basis"],
        charge=m.get("charge", 0),
        spin=m.get("spin", 0),
        unit="angstrom",
        verbose=0,
    )

    mf = scf.RHF(mol).run()

    active_space = m.get("active_space")
    cas = (active_space["n_electrons"], active_space["n_orbitals"]) if active_space else (mol.nelectron, mol.nao)

    return mol, mf, cas

def get_problem(cfg=None):
    """Qiskit route (EfficientSU2, UCCSD): ElectronicStructureProblem in the active space."""
    m = (cfg or get_config())["molecule"]
    problem = PySCFDriver(
        atom=m["geometry"],
        basis=m["basis"],
        charge=m.get("charge", 0),
        spin=m.get("spin", 0),
    ).run()
 
    active_space = m.get("active_space")
    if active_space:
        problem = ActiveSpaceTransformer(
            num_electrons=active_space["n_electrons"],
            num_spatial_orbitals=active_space["n_orbitals"],
            active_orbitals=active_space.get("active_orbitals"),
        ).transform(problem)
    return problem

def get_ansatz(name):
    """Returns (ansatz, qubit_hamiltonian, x0) for one ansatz."""
    mapper = JordanWignerMapper()
    problem = get_problem()
    n_orbs = problem.num_spatial_orbitals
    n_particles = problem.num_particles

    if name == "EfficientSU2":
        su2 = get_config()["ansatze"].get("EfficientSU2", {})
        ansatz = efficient_su2(
            2 * n_orbs,
            reps=su2.get("reps", 1),
            entanglement=su2.get("entanglement", "linear"),
        )
        hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

    elif name == "UCCSD":
        hf_state = HartreeFock(n_orbs, n_particles, mapper)
        ansatz = UCCSD(n_orbs, n_particles, mapper, initial_state=hf_state)
        hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
        x0 = np.zeros(ansatz.num_parameters)

    elif name == "fUCCSD":
        molecule, mf, cas = get_molecule()
        with contextlib.redirect_stdout(io.StringIO()):
            QI = QuantumInterface(StatevectorSampler(), "fUCCSD", mapper)
            wf = WaveFunctionCircuit(cas, mf.mo_coeff, molecule, QI)
        ansatz = QI.circuit
        n_act = wf.num_active_orbs
        h_ferm = wf._get_hamiltonian().get_qiskit_form(n_act)
        hamiltonian = mapper.map(FermionicOp(h_ferm, 2 * n_act))
        x0 = np.zeros(ansatz.num_parameters)

    else:
        raise ValueError(f"Unknown ansatz: {name}")

    return ansatz, hamiltonian, x0