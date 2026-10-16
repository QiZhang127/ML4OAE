# One LHS row per paired control/OAE simulation. RUN_MODE defaults to a three-day test.
using NumericalOcean, NumericalEarth, Oceananigans, OceanBioME
using Oceananigans.Units
using Oceananigans: fields
using Oceananigans.Biogeochemistry: biogeochemical_auxiliary_fields
using Oceananigans.Grids: inactive_node
using OceanBioME: PO₄, Fe
using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: primary_production
using CSV, DataFrames, CUDA, Dates, TOML

row_number = parse(Int, isempty(ARGS) ? get(ENV, "SLURM_ARRAY_TASK_ID", "1") : ARGS[1])
run_mode = get(ENV, "RUN_MODE", "test")
run_mode in ("test", "formal") || error("RUN_MODE must be test or formal.")
parse(Int, get(ENV, "SLURM_NTASKS", "1")) == 1 || error("Use one Slurm task and one GPU.")

LHS_file = joinpath(@__DIR__, "data", "parameters.csv")
parameter_table = CSV.read(LHS_file, DataFrame)
nrow(parameter_table) == 100 || error("Expected the existing 100-row LHS design.")
1 <= row_number <= nrow(parameter_table) || error("LHS row must be between 1 and 100.")
p = NamedTuple(parameter_table[row_number, :])
parameter_names = (:maximum_growth_rate, :chlorophyll_ratio, :iron_half_saturation,
                   :nitrate_half_saturation, :light_half_saturation, :maximum_grazing_rate)
all(name -> isfinite(p[name]) && p[name] > 0, parameter_names) || error("Invalid parameter values.")

# CSV units: day⁻¹, mg Chl (mmol N)⁻¹, nM Fe, µM N, W m⁻², day⁻¹.
plankton_parameters = (
    phytoplankton_maximum_growth_rate = p.maximum_growth_rate / 86400,
    chlorophyll_ratio = p.chlorophyll_ratio,
    nutrient_half_saturations = (
        nitrate = p.nitrate_half_saturation,
        ammonia = 0.001,
        iron = p.iron_half_saturation / 1000,
    ),
    light_half_saturation = p.light_half_saturation,
    maximum_grazing_rate = p.maximum_grazing_rate / 86400,
)

CUDA.functional() || error("No CUDA GPU is available. Submit this script to a GPU node.")
run_root = joinpath(@__DIR__, "model_runs")
mkpath(run_root)
job_id = get(ENV, "SLURM_JOB_ID", "local")
run_dir = mktempdir(run_root; prefix = "$(p.sample_id)_job$(job_id)_", cleanup = false)
CSV.write(joinpath(run_dir, "parameters.csv"), parameter_table[[row_number], :])

@info "LHS control/OAE" sample_id=p.sample_id row=row_number run_mode run_dir
@info "Selected parameters in CSV units" p
@info "Active environment" project=Base.active_project() NumericalOcean=pathof(NumericalOcean) OceanBioME=Base.pkgversion(OceanBioME)

arch = GPU()
grid = ORCA1(arch)
plankton = PhytoZoo(eltype(grid); plankton_parameters...)

# Retain the nitrogen-pool positivity modifier used by the previous LOBSTER runs.
nitrogen_positivity = ScaleNegativeTracers(
    (:NO₃, :NH₄, :P, :Z, :DOM, :sPOM, :bPOM); warn = false)

biogeochemistry = LOBSTER(grid;
    limiting_nutrients = (:nitrate, :ammonia, :iron),
    plankton,
    nutrients = Nutrients(nitrogen = NitrateAmmonia{eltype(grid)}(), phosphate = PO₄, iron = Fe),
    inorganic_carbon = CarbonateSystem(2),
    oxygen = Oxygen(eltype(grid)),
    light_attenuation = TwoBandPhotosyntheticallyActiveRadiation(grid, PARFromShortwave(grid)),
    scale_negatives = false,
    modifiers = nitrogen_positivity,
)

# Multi-year forcing and atmospheric CO₂ share the same calendar origin.
start_date = DateTime(1990, 1, 1)
# The exclusive stop date includes the complete 2010 calendar year.
formal_end_date = DateTime(2011, 1, 1)
# This is the forcing-record end, not the simulation stop date.
forcing_end_date = DateTime(2018, 1, 1)
pCO₂ = MaunaLoaCO₂(arch; start_date, end_date = forcing_end_date)

# Jago's Amazon release: 1 Tmol over 365 days, beginning on 2000-01-01.
# Model time is measured from start_date, including leap days.
oae_start_date = DateTime(2000, 1, 1)
oae_start_time = Dates.value(oae_start_date - start_date) / 1000
oae_center = (-49.5, 0.0)
oae_total_injection = 1e15 # mmol alkalinity = 1 Tmol
oae_duration = 365days
oae_sigma = 250_000.0 # m; Gaussian cutoff is three standard deviations
alkalinity_release = GaussianInjection(grid, oae_center, oae_start_time;
    total_injection = oae_total_injection, duration = oae_duration, σ = oae_sigma)

simulation = forced_ocean_simulation(grid;
    biogeochemistry,
    atmosphere_tracers = (; pCO₂),
    forcing = (; Alk2 = alkalinity_release),
    start_date,
    end_date = forcing_end_date,
    progress_frequency = 10,
)

# Load Jago's physical spin-up. This driver starts BGC from initial fields.
restart_path = pickup_restart!(simulation, start_date; start_date)
ocean = simulation.model.ocean.model
grid = ocean.grid

# Depth-dependent P/Z seeds inherited from Initialization_Ice_iron.jl (mmol N m⁻³).
# Nutrients, oxygen, DIC and alkalinity were initialized from UnifiedBGC.
upper_ocean_weight(z) = 0.5 * (1 + tanh((z + 100) / 30))
P_initial(x, y, z) = 0.001 + (0.01 - 0.001) * upper_ocean_weight(z)
Z_initial(x, y, z) = 0.0001 + (0.001 - 0.0001) * upper_ocean_weight(z)
set!(ocean; P = P_initial, Z = Z_initial, DOM = 0, sPOM = 0, bPOM = 0)

initial_time = simulation.model.clock.time
simulation.stop_time = if run_mode == "test"
    initial_time + 3days
else
    Dates.value(formal_end_date - start_date) / 1000
end
simulation.stop_time > initial_time || error("Stop time must exceed the restored clock time.")
initial_time < oae_start_time || error("The physical restart must precede the OAE release.")
@info "OAE schedule" start_date=oae_start_date duration_days=oae_duration/days total_Tmol=oae_total_injection/1e15

# Growth-based carbon production, including exudate, integrated over the full water column.
# The diagnostic is not a complete autotrophic-respiration budget.
@inline function carbon_production(i, j, k, grid, plankton, bgc, tracers, auxiliary)
    inactive_node(i, j, k, grid, Center(), Center(), Center()) && return zero(eltype(grid))
    return primary_production(i, j, k, grid, plankton, bgc, tracers, auxiliary) *
           convert(eltype(grid), 12.011 * 86400)
end

bgc = ocean.biogeochemistry.underlying_biogeochemistry
CHL = Field(bgc.plankton.chlorophyll_ratio * ocean.tracers.P)
production = KernelFunctionOperation{Center, Center, Center}(
    carbon_production, grid, bgc.plankton, bgc, fields(ocean),
    biogeochemical_auxiliary_fields(ocean.biogeochemistry))
PP = Field(Integral(production; dims = 3))

interface = simulation.model.interfaces.properties
control_surface, control_integrals = NumericalOcean.carbonate_replicate_diagnostics(
    (DIC = :DIC1, Alk = :Alk1, suffix = "1"), ocean, interface)
oae_surface, oae_integrals = NumericalOcean.carbonate_replicate_diagnostics(
    (DIC = :DIC2, Alk = :Alk2, suffix = "2"), ocean, interface)

# fCO2 is the control target for observational calibration.
fCO2 = control_surface.fCO₂1
fCO2_OAE = oae_surface.fCO₂2
CO2_flux_control = control_surface.surface_carbon_flux1
CO2_flux_OAE = oae_surface.surface_carbon_flux2
# Native gas flux is positive out of the ocean; this difference is positive for extra uptake.
additional_CO2_uptake = Field(CO2_flux_control - CO2_flux_OAE)

surface_outputs = (; CHL, fCO2, fCO2_OAE,
    CO2_flux_control, CO2_flux_OAE, additional_CO2_uptake,
    pH_control = control_surface.pH1, pH_OAE = oae_surface.pH2,
    P = ocean.tracers.P, Z = ocean.tracers.Z,
    T = ocean.tracers.T, S = ocean.tracers.S,
    DIC1 = ocean.tracers.DIC1, Alk1 = ocean.tracers.Alk1,
    DIC2 = ocean.tracers.DIC2, Alk2 = ocean.tracers.Alk2)
surface_writer = JLD2Writer(ocean, surface_outputs;
    including = Symbol[], indices = (:, :, grid.Nz),
    schedule = AveragedTimeInterval(1days; stride = 1),
    dir = run_dir, filename = "surface_daily", overwrite_files = false)
PP_writer = JLD2Writer(ocean, (; PP);
    including = Symbol[], indices = (:, :, 1),
    schedule = AveragedTimeInterval(1days; stride = 1),
    dir = run_dir, filename = "PP_daily", overwrite_files = false)
add_output_writer!(simulation, :targets_surface, surface_writer)
add_output_writer!(simulation, :targets_PP, PP_writer)

# Daily global inventories and gas-exchange rates support the OAE carbon budget.
integral_outputs = (
    DIC_control = control_integrals.DIC1, DIC_OAE = oae_integrals.DIC2,
    Alk_control = control_integrals.Alk1, Alk_OAE = oae_integrals.Alk2,
    CO2_flux_control = control_integrals.surface_carbon_flux1,
    CO2_flux_OAE = oae_integrals.surface_carbon_flux2,
    additional_CO2_uptake = Integral(additional_CO2_uptake),
)
integral_writer = JLD2Writer(ocean, integral_outputs;
    including = Symbol[], indices = (1, 1, 1), array_type = Array{Float64},
    schedule = AveragedTimeInterval(1days; stride = 1),
    dir = run_dir, filename = "carbon_integrals_daily", overwrite_files = false)
add_output_writer!(simulation, :carbon_integrals, integral_writer)
checkpointer!(simulation; output_dir = run_dir, filename_prefix = "paired",
    checkpoint_interval = run_mode == "test" ? 1days : 360days)

metadata = Dict(
    "sample_id" => p.sample_id, "lhs_row" => row_number, "run_mode" => run_mode,
    "lhs_file" => LHS_file, "restart_path" => restart_path,
    "calendar_origin" => string(start_date),
    "actual_start_date" => string(start_date + Millisecond(round(Int, initial_time * 1000))),
    "stop_date" => string(start_date + Millisecond(round(Int, simulation.stop_time * 1000))),
    "NumericalOcean_version" => string(Base.pkgversion(NumericalOcean)),
    "OceanBioME_version" => string(Base.pkgversion(OceanBioME)),
    "biomass_initialization" => "Inherited P/Z depth profiles; DOM, sPOM, bPOM zero",
    "CHL_units" => "mg Chl m-3", "PP_units" => "mg C m-2 day-1", "fCO2_units" => "uatm",
    "PP_definition" => "Full-column growth-based carbon production, including exudate",
    "calibration_fCO2" => "fCO2 (control; DIC1/Alk1)",
    "surface_CO2_flux_units" => "mmol C m-2 s-1; positive ocean to atmosphere",
    "global_CO2_flux_units" => "mmol C s-1; positive ocean to atmosphere",
    "additional_CO2_uptake_definition" => "Control gas flux minus OAE gas flux; positive for extra uptake",
    "global_inventory_units" => "DIC: mmol C; Alk: mmol alkalinity",
    "carbonate_replicates" => "1: control; 2: OAE; shared CHL and PP",
    "oae" => Dict(
        "longitude" => oae_center[1], "latitude" => oae_center[2],
        "start_date" => string(oae_start_date),
        "end_date" => string(oae_start_date + Millisecond(round(Int, oae_duration * 1000))),
        "duration_days" => oae_duration / days,
        "total_alkalinity_Tmol" => oae_total_injection / 1e15,
        "sigma_m" => oae_sigma, "cutoff_m" => 3 * oae_sigma,
        "forced_tracer" => "Alk2"),
    "csv_parameters" => Dict(string(name) => p[name] for name in parameter_names),
    "model_parameters" => Dict(
        "growth_s-1" => plankton.phytoplankton_maximum_growth_rate,
        "chlorophyll_ratio" => plankton.chlorophyll_ratio,
        "iron_mmol_m-3" => plankton.nutrient_half_saturations.iron,
        "nitrate_mmol_m-3" => plankton.nutrient_half_saturations.nitrate,
        "light_W_m-2" => plankton.light_half_saturation,
        "grazing_s-1" => plankton.maximum_grazing_rate),
)
open(joinpath(run_dir, "run_metadata.toml"), "w") do io
    TOML.print(io, metadata)
end

try
    run!(simulation)
finally
    flush_output!(simulation)
end
@info "Paired control/OAE run completed" sample_id=p.sample_id run_dir
