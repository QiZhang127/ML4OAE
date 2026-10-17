# ML4OAE

Neural network emulators for biogeochemical parameter calibration and ocean alkalinity enhancement (OAE) experiments with [NumericalOcean](https://github.com/jagoosw/NumericalOcean).

The emulators predict chlorophyll (CHL), column-integrated primary production (PP), and surface CO₂ fugacity (fCO2) from six biological parameters. SVD and neural network methods follow [Dagon et al. (2020)](https://ascmo.copernicus.org/articles/6/223/2020/).

## Status

Pilot analysis uses four simulations from 2000–2001. The 100-member simulation setup is under testing; formal calibration and validation remain pending.

## Files

| File | Purpose |
|---|---|
| `oae_nn_emulator.ipynb` | LHS design, SVD, NN tuning, evaluation, and calibration |
| `prepare_pilot_2000_2001.ipynb` | Export the existing Julia/JLD2 pilot outputs |
| `run_lhs_control.jl` | Run one paired control/OAE member |
| `submit_lhs.sbatch` | Slurm submission template |

## Setup

```bash
python -m pip install -r requirements.txt
```

The notebooks locate the repository from the working directory. Paths can be set in their configuration cells or through these environment variables:

| Variable | Default location |
|---|---|
| `ML4OAE_DIR` | Repository directory |
| `MODEL_ROOT` | Sibling `global-ocean-model` directory for the pilot exporter |
| `OBSERVATION_DIR` | Sibling `ocean_observations` directory |
| `NUMERICALOCEAN_DIR` | Sibling `NumericalOcean` Julia project for simulations |

The submission script preserves an existing `JULIA_DEPOT_PATH`; otherwise it uses the sibling `julia_depot` directory. Dependency installation and submission must use the same depot. Slurm account, resources, node exclusions, and module names are cluster-specific settings in the template.

Submit the three-day setup test from the repository root:

```bash
sbatch --export=ALL,RUN_MODE=test submit_lhs.sbatch 1
```

## Analysis

For the pilot, run the Julia export notebook, then the Python notebook with `run_mode = "pilot"`. The exporter requires the original model environment, including OceanBioME v0.18.0.

Formal analysis uses `run_mode = "formal"`, 100 completed controls, recorded parameters, and `data/prepared_data.npz`. Calibration covers **2000–2010**, with model and observation fields matched in grid, units, period, and diagnostic definition. Observations use Copernicus CHL/PP and GCB ensemble fCO2. New ORCA1 outputs require separate preprocessing from the pilot export.

Each target has its own SVD and NN. Tuning compares 121 architectures, then evaluates the leading five across 100 training/validation splits. Formal calibration combines a 10-million-member candidate search and differential evolution using a joint observation cost. Proposed parameters require new ocean-model runs for validation against the baseline and observations.

## Experiments and outputs

The Julia driver uses ORCA1, LOBSTER, and JRA55 forcing from a selected 1990 physical restart through 2010. BGC tracers start from initialized fields and require adjustment checks. The Amazon OAE release adds 1 Tmol alkalinity over 365 days from 2000-01-01. Paired carbonate systems share CHL and PP; calibration uses control fCO2.

`results/` holds fitted networks, preprocessing arrays, tuning summaries, metrics, plots, and parameter proposals. `model_runs/` holds simulation fields, checkpoints, and metadata.
