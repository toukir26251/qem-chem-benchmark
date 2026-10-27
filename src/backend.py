"""Fake backend, estimators, sampler, transpiling and VQE (same setup as the notebooks)."""

from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_ibm_runtime import fake_provider

from .utils import get_config


def get_backend():
    """Returns the fake backend named in the config (e.g. FakeTorino)."""
    return getattr(fake_provider, get_config()["backend"]["name"])()


def get_transpiled(circuit, parameters, hamiltonian):
    """Binds the parameters, transpiles for the backend and maps the Hamiltonian to the layout."""
    seed = get_config()["vqe"]["seed"]
    bound = circuit.assign_parameters(parameters)
    transpiled = transpile(bound, get_backend(), seed_transpiler=seed)
    return transpiled, hamiltonian.apply_layout(transpiled.layout)


def get_ideal_estimator():
    """Returns a noiseless Aer estimator (for exact energies of transpiled circuits)."""
    return EstimatorV2(options={
        "backend_options": {"method": "statevector"},
        "default_precision": 0.0,
    })


def get_noisy_estimator(seed=None):
    """Returns a noisy Aer estimator with the backend's noise model.

    mode: precision -> default_precision from the config
    mode: shots     -> default_precision = 1/sqrt(shots)
    """
    cfg = get_config()["backend"]
    seed = get_config()["vqe"]["seed"] if seed is None else seed

    if cfg["mode"] == "precision":
        precision = cfg["precision"]
    elif cfg["mode"] == "shots":
        precision = 1 / cfg["shots"] ** 0.5
    else:
        raise ValueError(f"backend.mode must be 'precision' or 'shots', got '{cfg['mode']}'")

    return EstimatorV2(options={
        "backend_options": {
            "method": cfg.get("method", "density_matrix"),
            "noise_model": NoiseModel.from_backend(get_backend()),
        },
        "default_precision": precision,
        "run_options": {"seed_simulator": seed},
    })


def get_noisy_sampler(seed=None):
    """Returns a noisy Aer sampler (needed for SlowQuant / M0)."""
    cfg = get_config()["backend"]
    seed = get_config()["vqe"]["seed"] if seed is None else seed
    return SamplerV2(
        default_shots=cfg["shots"],
        seed=seed,
        options={"backend_options": {"noise_model": NoiseModel.from_backend(get_backend())}},
    )


def run_vqe(ansatz, hamiltonian, x0):
    """Runs noiseless VQE and returns energy (electronic) and optimized parameters."""
    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=SLSQP(maxiter=get_config()["vqe"]["maxiter"]),
        initial_point=x0,
    )
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    return result.eigenvalue.real, result.optimal_point