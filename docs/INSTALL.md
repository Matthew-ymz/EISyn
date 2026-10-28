# Installation and software citation

Use Python 3.11 and run the commands below from the repository root. The
repository currently uses scripts, local modules and notebooks; keep the cloned
source tree available when running them.

## Install with pip

```bash
git clone https://github.com/Matthew-ymz/EISyn.git
cd EISyn
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead of the `source` command.

The direct dependency pins provide a reference environment. They include
PyTorch for learned transition models and xarray for climate data. They do not lock every transitive dependency or establish
that every historical experiment runs in this environment.

## Install with Conda

The Conda file selects Python 3.11 and reuses the same pip requirements:

```bash
conda env create -f environment.yml
conda activate eisyn
python -m pip check
```

## Optional dependencies

For tests and Jupyter notebooks:

```bash
python -m pip install -r requirements-dev.txt
python -m jupyter lab
```

For the additional data formats, brain-surface tools and comparison methods:

```bash
python -m pip install -r requirements-experiments.txt
python -m pip check
```

| Profile | Coverage |
|---|---|
| `requirements.txt` | Numerical computation, plotting, common machine-learning modules, PyTorch and xarray |
| `requirements-dev.txt` | Base profile plus pytest and notebook tools |
| `requirements-experiments.txt` | Base profile plus Parquet/NetCDF/HDF5, brain surfaces, SHAP, Tigramite and additional analyses |

Install both optional profiles if both are needed. The optional ranges are
installation constraints, not a frozen record of all paper experiments. Older
exploratory code may need additional upstream software. In particular, the
legacy `exp/brain/dmf_fig6.py` imports an external `phyid` implementation;
this is not supplied by the profiles above. The legacy `pickle5` import is also
not an installation requirement for the Python 3.11 environment.

## Check the installation

Check representative imports without downloading data or training a model:

```bash
python -c "import numpy, scipy, pandas, matplotlib, sklearn, torch, xarray, utils; from scripts.spt import build_spt; from exp.TM.transport_map_density import estimate_mutual_information_transport_map; print('EISyn imports OK')"
```

After installing the development profile, run the small deterministic checks:

```bash
python -m pytest -q tests/test_spt_core.py tests/test_greedy_hierarchy_known_dynamics.py
```

These checks cover tree construction, nonnegativity handling and a discrete
known-dynamics example. They do not reproduce all manuscript results. A CPU is
sufficient for these checks. Full neural-network training and application
analyses have separate data, memory and runtime requirements. Installation time
depends on the platform, connection and PyTorch wheel; no clean-machine install
time has been benchmarked yet.

## Data, models and experiment instructions

Dependency installation does not download the local `data/`, `models/`,
`results/` or `fig/` contents. Some are excluded from Git. Start with the
[reproduction index](reports/README.md) for the existing calculation and
plotting entry points. UniCM analyses additionally use upstream source code
and checkpoints under `data/UniCM-checkpoint/`; see the
[climate entry points](reports/README.md#climate). Brain and NYC Taxi resources
are indexed under [Brain](reports/README.md#brain) and
[urban mobility](reports/README.md#urban-mobility).

The optional profiles do not install upstream model repositories or grant
access to restricted datasets. A complete paper release still needs a frozen
figure-to-code index and accessible data/model artifacts.

## License and citation

Repository-authored software is available under the [MIT License](../LICENSE).
Third-party code, pretrained weights and datasets retain their respective
licenses and access conditions.

[CITATION.cff](../CITATION.cff) provides software citation metadata. Cite the
commit or release actually used. A release version, release date and DOI have
not been assigned in this file; add the real archive metadata when the paper
code is released. Software credit is currently recorded collectively as
**EISyn contributors**. The manuscript author list is separate and should be
confirmed for the final paper citation.
