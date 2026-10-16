# ML4OAE

Neural network emulators for biogeochemical parameter calibration and ocean alkalinity enhancement (OAE) experiments with [NumericalOcean](https://github.com/jagoosw/NumericalOcean).

Six biological parameters are mapped to chlorophyll (CHL), column-integrated primary production (PP), and surface CO₂ fugacity (fCO2). The approach combines singular value decomposition (SVD) and neural networks, following [Dagon et al. (2020)](https://ascmo.copernicus.org/articles/6/223/2020/).

## Status

The emulator has been tested on four existing simulations covering 2000–2001. The simulation setup for the 100-member ensemble is under testing. Formal calibration and validation remain pending.

## Files

| File | Purpose |
|---|---|
| `oae_nn_emulator.ipynb` | LHS design, SVD, NN tuning, evaluation, and parameter calibration |
| `prepare_pilot_2000_2001.ipynb` | Export existing pilot simulations using Julia |
| `run_lhs_control.jl` | Run one paired control/OAE ensemble member |
| `submit_lhs.sbatch` | Submit simulations through Slurm |

## Usage

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

For the pilot analysis, run the Julia export notebook first, then the Python emulator notebook with `run_mode = "pilot"`.

Formal analysis requires 100 completed simulations, their recorded parameter values, and matched model and observation fields in `data/prepared_data.npz`. The pilot exporter handles the older model outputs; new ORCA1 outputs require separate preprocessing. Formal notebook settings are:

```python
run_mode = "formal"
period = ("2000-01-01", "2010-12-31")
```

Model and observation fields must share the same period, grid, units, and diagnostic definitions. Observations use Copernicus CHL/PP and GCB ensemble fCO2. Data locations and cluster settings are configured in the notebooks and submission script.

## Workflow

1. Generate 100 six-parameter LHS combinations and run the ocean model for each member.
2. Fit separate SVD bases and neural networks for CHL, PP, and fCO2.
3. Compare 121 two-hidden-layer architectures, then evaluate the leading five over 100 training/validation splits. Select each target's architecture by mean validation MSE; test cases remain separate.
4. In formal mode, evaluate 10 million candidate parameter combinations and apply differential evolution using a joint observation-based cost.
5. Validate the proposed parameters with new ocean-model simulations against the baseline and observations.

## Simulation configuration

The Julia driver uses ORCA1, LOBSTER, and JRA55 forcing. Formal runs cover the selected 1990 physical restart through 2010. BGC tracers start from initialized fields and require adjustment checks.

Paired carbonate systems represent control and OAE conditions. A 1 Tmol alkalinity release begins on 2000-01-01 at the Amazon mouth and lasts 365 days. CHL and PP are shared between the carbonate systems; calibration uses control fCO2.

The three-day setup test is submitted with:

```bash
sbatch --export=ALL,RUN_MODE=test submit_lhs.sbatch 1
```

## Outputs

`results/` contains trained networks, preprocessing arrays, tuning summaries, evaluation metrics, and plots. Formal calibration also produces candidate costs and `proposed_parameters.csv`. `model_runs/` contains simulation fields, checkpoints, and run metadata.
