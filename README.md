# OAE neural-network emulator

Emulate CHL, column-integrated PP, and surface fCO2 from six biological parameters. The workflow adapts [Dagon et al. (2020)](https://ascmo.copernicus.org/articles/6/223/2020/): **100-member LHS design → control simulations → SVD → NN training and testing → observation cost → 10-million-member LHS exploration → differential evolution → physical-model validation**.

`oae_nn_emulator.ipynb` is the main notebook for both pilot and formal analysis.

## Setup

Place the files together in `ML4OAE` and install the Python requirements:

```bash
python -m pip install -r requirements.txt
```

`prepare_pilot_2000_2001.ipynb` runs in the model's Julia environment with JLD2, Oceananigans, OceanBioME v0.18.0, and NCDatasets. `oae_nn_emulator.ipynb` runs in Python, from the project folder or its parent. Source paths, case folders, dates, and project location are configured in the export notebook's first cell.

The Python notebook has two modes, selected by `run_mode` in Section 1:

| Mode | Inputs | Purpose |
|---|---|---|
| `"pilot"` | Existing 2000–2001 control cases | Test SVD and NN predictions; compare existing cases with observations |
| `"formal"` | 100 completed LHS controls and matched observations | Calibrate all three targets over 2000–2010 |

Keep the pilot dates matched to the existing model export. For formal calibration, set the period in Section 1 to:

```python
period = pilot_period if is_pilot else ("2000-01-01", "2010-12-31")
```

## Pilot

1. Run `prepare_pilot_2000_2001.ipynb` with a Julia kernel. Set `model_root`, `project_dir`, `cases`, and the period to match the control outputs.
2. Check the generated **`data/pilot_parameters.csv`**. Metadata provides the six parameter values. Missing Reference chlorophyll and light parameters use `reference_defaults` (1.31 and 33.0); other missing entries require values from the run configuration. Existing entries are retained and checked against metadata. Newly selected cases are added from metadata. Only selected cases are exported; other CSV rows may remain.
3. The export produces **`data/pilot_2000-2001.nc`** with one period-mean field per independent case. Continuation segments are joined to their original case. All selected cases must cover the same complete period and use compatible model settings.
4. Run `oae_nn_emulator.ipynb` in Python with `run_mode = "pilot"`. Set `pilot_period` to the export dates. Results are saved in `results/pilot_2000-2001/`.

CHL and full-column PP use saved diagnostics. Control fCO2 is calculated from surface DIC1, Alk1, temperature, and salinity with OceanBioME carbonate chemistry, then averaged in time. This reconstruction can take longer than NN training. Missing required inputs stop the export.

The pilot requires at least four distinct parameter cases. With four cases, the training/validation/test split is 2/1/1; retained SVD modes are limited by training rank. Pilot metrics describe this small split, not performance across the full six-parameter range. Observational calibration is reserved for formal mode.

Section 7b compares the actual pilot fields with downloaded Copernicus CHL/PP and GCB ensemble fCO2 over 2000–2001. It reads the sibling `ocean_observations` folder, interpolates monthly observations to model cell centres and weights them by calendar-month lengths. Each target uses cells with all 24 monthly observations and finite values in every case. Missing observations remain missing. This is bilinear point sampling, not conservative area remapping.

`case_vs_observations.csv` reports area-weighted RMSE, model-minus-observation bias, valid cell count and area coverage. The lowest RMSE identifies the closest existing case separately for each target; RMSEs with different units are not added. This comparison does not perform six-parameter calibration. Pilot mode skips the large candidate search and differential evolution in Section 8.

## LHS design

Sections 1–2 of the Python notebook save **100 parameter sets** in `data/parameters.csv`, with IDs `sample_0001` through `sample_0100`. This design is separate from the pilot parameter table.

Sampling uses `scipy.stats.qmc.LatinHypercube(d=6, scramble=True, optimization="random-cd", seed=42)`. Each parameter has one sample in each of 100 equal intervals, scaled linearly to its bounds. Method, bounds, units, seed, and SciPy version are stored in `data/lhs_settings.json`. Existing design files are reused unchanged.

Run one physical control simulation per design row and retain its ID and parameter values. The baseline control is a separate comparison run.

## Formal inputs

Set `run_mode = "formal"` after preparing the 100 completed controls and matched observations. All three targets use **2000–2010**. Simulations may cover 1990–2010, but the fields supplied for calibration must be averaged over 2000–2010. Do not compare model and observation means from different periods.

`data/parameters.csv` contains `sample_id` and these six columns:

| Column | Unit | Bounds |
|---|---|---|
| `maximum_growth_rate` | day^-1 | 0.4–3.3 |
| `chlorophyll_ratio` | mg Chl (mmol N)^-1 | 0.26–2.62 |
| `iron_half_saturation` | nM | 0.02–1.00 |
| `nitrate_half_saturation` | uM | 0.05–3.00 |
| `light_half_saturation` | W m^-2 | 10–100 |
| `maximum_grazing_rate` | day^-1 | 0.2–1.8 |

`data/prepared_data.npz` contains a string `sample_id` array and these arrays for each target `CHL`, `PP`, and `fCO2`:

| Key, using CHL as an example | Shape | Content |
|---|---|---|
| `CHL` | `(100, n_features)` | Actual control mean fields for 2000–2010 |
| `CHL_obs_mean` | `(n_features,)` | Observation mean for 2000–2010 |
| `CHL_obs_years` | `(11, n_features)` | Annual observation means, 2000 through 2010 |
| `CHL_area` | `(n_features,)` | Positive cell areas |

CSV and NPZ rows are matched by ID. Use recorded parameters from the actual runs; do not regenerate a design after running the simulations. Each target may use a separate grid, but its model and observation features must match in order, units, depth and period. CHL uses `mg Chl m^-3`, PP uses `mg C m^-2 day^-1`, and fCO2 uses `uatm`. PP must have a consistent net/gross and column-integration definition; fCO2 must not be replaced by pCO2.

Prepare observation means and annual fields from the actual products under `ocean_observations`: Copernicus CHL and PP in `raw/copernicus_ocean_colour/`, and variable `sfco2` in `subsets/GCB2025_fCO2_ensemble_mean_1990-2010.nc`. Select only 2000–2010 for formal calibration. The pilot arrays created in Section 7b cover 2000–2001 and do not replace these formal inputs.

Formal NPZ preparation is separate from the pilot export. Missing observations must remain missing, with a consistent mask across the observation mean, annual fields and control fields. The SVD fitting support uses finite observations and training simulations; all held-out simulations must also be finite on that support. Raw file downloads alone do not create `prepared_data.npz`.

## Running paired LHS experiments

`run_lhs_control.jl` reads one row of `data/parameters.csv` and builds a paired control/OAE experiment on NumericalOcean's ORCA1 configuration with LOBSTER. It uses multi-year JRA55 forcing and the physical restart selected by NumericalOcean. The six sampled parameters are applied to `PhytoZoo`; growth and grazing rates are converted from day^-1 to s^-1, and iron from nM to mmol m^-3.

`CarbonateSystem(2)` carries `DIC1`/`Alk1` for the control and `DIC2`/`Alk2` for OAE. A single 365-day release of **1 Tmol alkalinity** begins on **2000-01-01** at the Amazon mouth (**49.5 degrees W, 0 degrees N**), applied to `Alk2` in the surface cells. The Gaussian footprint has a 250 km standard deviation and 750 km cutoff, matching [NumericalOcean's release defaults](https://github.com/jagoosw/NumericalOcean/blob/main/src/alkalinity_injection.jl). Calendar dates set the start at day 3652 since 1990-01-01. The 365-day release ends on 2000-12-31; 2000 is a leap year.

The driver uses UnifiedBGC for supported tracers. It retains the depth-dependent P/Z seed profiles and nitrogen-pool positivity modifier from the earlier LOBSTER configuration, with DOM and particulate organic pools initially zero. These initial fields are not a spun-up BGC restart. The nearest physical restart date may differ from 1990-01-01; the actual date is recorded.

Keep `README.md`, `run_lhs_control.jl` and `submit_lhs.sbatch` together in `/nfs/roberts/pi/pi_ey239/qz334/ML4OAE/`, alongside the emulator notebook. The relevant layout is:

```text
qz334/
├── ML4OAE/
│   ├── README.md
│   ├── oae_nn_emulator.ipynb
│   ├── run_lhs_control.jl
│   ├── submit_lhs.sbatch
│   ├── data/
│   │   └── parameters.csv
│   └── model_runs/
└── NumericalOcean/
    └── Project.toml
```

The submission script runs from `ML4OAE` and uses the Julia environment in the sibling `NumericalOcean` directory. Parameter input and simulation output stay under `ML4OAE`; `model_runs/` is created automatically. Instantiate the updated NumericalOcean project once:

```bash
module load Julia
julia --project=/nfs/roberts/pi/pi_ey239/qz334/NumericalOcean -e 'using Pkg; Pkg.instantiate()'
cd /nfs/roberts/pi/pi_ey239/qz334/ML4OAE
sbatch --export=ALL,RUN_MODE=test submit_lhs.sbatch 1
```

The default run advances row 1 by three model days after loading the physical restart, before the OAE release. It writes daily surface CHL and fCO2, full-column PP, supporting surface tracers, daily checkpoints, the selected CSV row and `run_metadata.toml`. Each submission has a separate directory under `model_runs/`, for example `sample_0001_job1234567_aBc123/`; its full path is printed in the job log. Directory names contain the sample ID, numeric Slurm job ID and a random suffix. The three-day run checks setup and output, not biogeochemical adjustment or active OAE injection.

`surface_daily.jld2` contains `fCO2` for the control, `fCO2_OAE`, both carbonate tracer pairs, pH, both air-sea CO2 fluxes and their difference. Fluxes are in mmol C m^-2 s^-1 and positive from ocean to atmosphere; `additional_CO2_uptake` is control minus OAE flux, positive for extra atmospheric uptake. `carbon_integrals_daily.jld2` saves daily global carbonate inventories and gas-exchange rates in double precision. Integrate its `additional_CO2_uptake` rate over time and divide by 1e15 to express additional uptake in Tmol C. `PP_daily.jld2` contains full-column PP.

CHL and PP are shared by the two carbonate systems. This setup measures carbonate-chemistry and gas-exchange responses under biological parameter uncertainty; it has no alkalinity- or pH-dependent feedback on phytoplankton growth. Prepare emulator calibration fields from CHL, PP and control `fCO2`.

Here PP is growth-based carbon production including exudate, converted to mg C m^-2 day^-1 and integrated over the full column. It does not include a separate autotrophic-respiration budget. Confirm the diagnostic's compatibility with the observational PP definition before calibration.

After checking the short run, `RUN_MODE=formal` runs from the selected 1990 physical restart to 2011-01-01, including the OAE release and the complete 2010 calendar year. Run the first complete member with:

```bash
sbatch --export=ALL,RUN_MODE=formal submit_lhs.sbatch 1
```

A job array selects all 100 rows using `SLURM_ARRAY_TASK_ID`:

```bash
sbatch --array=1-100%1 --export=ALL,RUN_MODE=formal submit_lhs.sbatch
```

`%1` runs one member at a time. Adjust concurrency and the six-hour time request after measuring a complete member. Check BGC adjustment before accepting the 2000–2010 calibration fields. Daily output includes the adjustment years. These ORCA1 outputs require their own preprocessing; the older pilot exporter is for a different model grid and output format.

## Training and outputs

Each target has its own area-weighted SVD and NN. The formal split is 60 training, 20 validation, and 20 test members. Centering, SVD, and score scaling use the training members. Retained variance and reconstruction errors are reported alongside NN test errors.

The initial setting retains three SVD modes. Inspect retained variance and held-out reconstruction errors to determine whether three modes adequately represent each target. A high retained fraction does not establish NN prediction accuracy.

The architecture search follows the range and resampling procedure reported by Dagon et al. (2020). Both modes compare all 121 two-hidden-layer configurations with 5–15 nodes per layer. The first hidden layer uses ReLU, the second uses tanh, and the output layer is linear. Training uses RMSprop (learning rate 0.005), batch size 30 and validation early stopping. The fixed Keras L2 coefficient (`l2_strength=0.001`) and formal epoch limit (500) follow the [author's NN code](https://github.com/katiedagon/CLM5_ParameterUncertainty/blob/main/NN/NN_test.py). Pilot mode uses a shorter 100-epoch limit. Early stopping monitors standardized validation MSE, uses patience of 15 epochs in pilot mode and 50 in formal mode, and restores the best weights.

The initial validation MSE ranks the architectures. The leading five are evaluated over 100 random training/validation splits, refitting SVD and score scaling in every split. Test members remain separate. Each split reports MSE and per-mode squared Pearson correlations. The final selection minimizes mean resampled validation MSE, with its standard deviation used to break ties; correlations are reported for inspection. This explicit ranking rule implements the paper's skill/stability assessment without assuming an unpublished formula.

The default search trains 622 networks per target, or 1,866 for all three targets. It runs in both pilot and formal modes. Four pilot cases provide only one validation case per split, so correlations are undefined and the selection remains preliminary. Repeated splits do not add independent cases.

`tuning.csv` records all architectures and resampling summaries; `tuning_resamples.csv` records individual repeats. `settings.json` records the search settings and selected architecture for each target. The two tuning tables are saved after each target completes.

## Observational calibration

Section 8 runs only in formal mode. Project the observation mean and annual fields into the corresponding model SVD space. The joint cost sums variance-weighted squared U-score differences across modes and all three targets, divided by the annual observation-score variance. This normalization represents observed interannual variability, not measurement uncertainty.

Generate one global six-dimensional Latin hypercube with **10 million members**, then predict scores and calculate costs in batches of 100,000. This design is separate from the 100-member model-training ensemble. Save the 1,000 lowest-cost parameter sets in `top_1000_candidates.csv`. Generating the full float64 design requires about 480 MB for the candidate array, with additional temporary memory and cost storage; batching limits prediction memory. Global maps are not reconstructed for every candidate.

Differential evolution independently searches the same six bounds using the same joint cost. Compare its result with the best LHS member and configured default, then retain the lowest-cost parameter set. `optimization_summary.csv` records the three costs; `proposed_parameters.csv` stores the selected six physical parameter values. If the default has the lowest cost, the searches have not identified a predicted improvement. Use the actual default parameter values in `parameters.baseline`.

The large LHS exploration and differential evolution are separate parts of the paper's workflow (Sections 4.2–4.3). Low-cost LHS members describe parameter regions; they are not posterior samples or confidence intervals. The numerical proposal requires physical-model validation.

## Physical-model validation

Run a new control simulation with the proposed parameters. Compare it with the actual baseline and observations using the same 2000–2010 period, grid, mask and diagnostic definitions. Section 10 reads the actual baseline and optimized mean fields from `data/validation_runs.npz`, using keys such as `CHL_baseline` and `CHL_optimized` for each target. It writes area-weighted observation RMSE to `actual_model_comparison.csv`.

The notebook does not submit or run the ocean model. This validation remains pending until the physical simulations are completed and their fields are supplied. OAE experiments follow control-model calibration and validation.

## Output files

Models (`CHL.keras`, `PP.keras`, `fCO2.keras`), preprocessing arrays, settings and metrics are saved under `results/pilot_2000-2001/` or `results/formal/`.

| File | Content |
|---|---|
| `tuning.csv`, `tuning_resamples.csv` | NN architecture comparisons and resampling results |
| `test_metrics.csv`, `test_predictions.png` | Emulator predictions compared with held-out simulations |
| `svd_spatial_modes.png` | Pilot SVD patterns |
| `observations_2000-2001.npz` | Pilot observation means from Section 7b |
| `case_vs_observations.csv` | Existing pilot cases compared with observations |
| `top_1000_candidates.csv` | Lowest-cost members of the formal large LHS |
| `optimization_summary.csv` | Default, best LHS and differential-evolution costs |
| `proposed_parameters.csv` | Suggested biological parameters for a physical-model run |
| `actual_model_comparison.csv` | Actual baseline/optimized simulations compared with observations |

Files appear only after their corresponding steps run. Git tracks small parameter tables and LHS settings; raw fields and results are ignored.
