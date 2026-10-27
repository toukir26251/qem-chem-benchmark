# QEM Chem Benchmark

A benchmark of quantum error mitigation (QEM) techniques for quantum-chemistry calculations on noisy hardware. The project runs VQE for small molecules (H₂, LiH) with several ansätze (EfficientSU2, UCCSD, fUCCSD, pp-tUPS) on a simulated IBM backend (`FakeTorino`) and compares how well zero-noise extrapolation (ZNE) and Clifford data regression (CDR), M0 from SlowQuant recover the exact ground-state energy.
Molecules, ansätze, backend and mitigation settings are set in YAML files under `configs/`. Results go to `data/` and `plots/`, and the notebooks in `notebooks/` walk through each benchmark. This repository is a part (work package 1) of my Master's thesis.

## Installation

**Requirements:** Python 3.11 or 3.12 (Mitiq does not support 3.13+ yet).

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/qem-chem-benchmark.git
   cd qem-chem-benchmark
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv env
   source env/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install SlowQuant (used for fUCCSD and pp-tUPS). It is not available on PyPI, so install it from source:

   ```bash
   git clone https://github.com/erikkjellgren/SlowQuant.git
   cd SlowQuant
   pip install -e .
   cd ..
   ```

5. *(Optional, for JupyterLab users)* Register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name qem-chem-benchmark
   ```

   Then select the `qem-chem-benchmark` kernel when opening the notebooks.

6. Check that everything works:

   ```bash
   python -c "import qiskit, mitiq, pyscf, slowquant; print('All set')"
   ```

## How to run

Activate the environment first (`source env/bin/activate`) and run everything from the repository root.

### Option 1: Script

`scripts/run_benchmark.py` takes a config file and benchmarks EfficientSU2, UCCSD and fUCCSD with ZNE and CDR:

```bash
python scripts/run_benchmark.py configs/h2.yaml
python scripts/run_benchmark.py configs/lih.yaml
```

The runs can take a long time (LiH especially). To run in the background and keep going after you close the terminal:

```bash
nohup python scripts/run_benchmark.py configs/lih.yaml &
```

Where the results go (`<date>` is the run date, e.g. `2026-10-27`):

| Output | Location |
| --- | --- |
| Text log (energies, errors) | `outputs/<molecule>_<date>.log` |
| Raw results | `data/<molecule>_<ansatz>_<ZNE or CDR>_<date>.npz` |
| Plots | `plots/` |

Follow a running job with `tail -f outputs/LiH_<date>.log`.

### Option 2: Notebooks

Start JupyterLab (or open the notebooks in VS Code) and select the `qem-chem-benchmark` kernel:

```bash
jupyter lab
```

- `notebooks/01_H2_benchmark.ipynb`: H₂ benchmark
- `notebooks/02_LiH_benchmark.ipynb`: LiH benchmark

Run the cells from top to bottom.

### Changing the settings

Edit the YAML file in `configs/` (`h2.yaml`, `lih.yaml`) to change the molecule geometry, active space, VQE optimizer, backend (`FakeTorino`, precision or shots) and the ZNE/CDR parameters. To add a new molecule, copy one of the files and adjust it.
