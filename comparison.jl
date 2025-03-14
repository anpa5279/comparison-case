using Pkg
using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, hour
using Oceananigans.BuoyancyFormulations: g_Earth

mutable struct Params
    Nx::Int
    Ny::Int     # number of points in each of horizontal directions
    Nz::Int     # number of points in the vertical direction
    Lx::Float64
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    amplitude::Float64 # m
    wavelength::Float64 # m
    τx::Float64 # m² s⁻², surface kinematic momentum flux
    Jᵇ::Float64 # m² s⁻³, surface buoyancy flux
    N²::Float64 # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 #m 
    Q::Float64      # W m⁻², surface _heat_ flux
    ρₒ::Float64     # kg m⁻³, average density at the surface of the world ocean
    cᴾ::Float64     # J K⁻¹ kg⁻¹, typical heat capacity for seawater
    dTdz::Float64   # K m⁻¹, temperature gradient
end

#defaults, these can be changed directly below
params = Params(32, 32, 32, 128.0, 128.0,64.0, 0.8, 60.0, -3.72e-5, 2.307e-8, 1.936e-5, 33.0, 200.0, 1026.0, 3991.0, 0.01)

# Automatically distributes among available processors

arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), extent=(params.Lx, params.Ly, params.Lz))
@show grid

#B_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(params.Jᵇ),
#                                bottom = GradientBoundaryCondition(params.N²))

buoyancy = SeawaterBuoyancy()

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(params.Q / (params.ρₒ * params.cᴾ)),
                                bottom = GradientBoundaryCondition(params.dTdz))

@show T_bcs

@inline Jˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹

const evaporation_rate = 1e-3 / hour # m s⁻¹

evaporation_bc = FluxBoundaryCondition(Jˢ, field_dependencies=:S, parameters= evaporation_rate)

S_bcs = FieldBoundaryConditions(top=evaporation_bc)
@show S_bcs

const wavenumber = 2π / params.wavelength # m⁻¹
const frequency = sqrt(g_Earth * wavenumber) # s⁻¹

# The vertical scale over which the Stokes drift of a monochromatic surface wave
# decays away from the surface is `1/2wavenumber`, or
const vertical_scale = params.wavelength / 4π

# Stokes drift velocity at the surface
const Uˢ = params.amplitude^2 * wavenumber * frequency # m s⁻¹

@inline uˢ(z) = Uˢ * exp(z / vertical_scale)

@inline ∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(params.τx))
@show u_bcs

coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs)) # , b=B_bcs
@show model

@inline Ξ(z) = randn() * exp(z / 4)

# Temperature initial condition: a stable density gradient with random noise superposed.
@inline Tᵢ(x, y, z) = 20 + params.dTdz * z + params.dTdz * model.grid.Lz * 1e-6 * Ξ(z)

#@inline stratification(z) = z < - params.initial_mixed_layer_depth ? params.N² * z : params.N² * (-params.initial_mixed_layer_depth)

#@inline bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * params.N² * model.grid.Lz

u★ = sqrt(abs(params.τx))
@inline uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
@inline wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, T=Tᵢ, S=35) #, b=bᵢ)

simulation = Simulation(model, Δt=45.0, stop_time = 4hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields_$rank.jld2",
                                                      overwrite_existing = true,
                                                      with_halos = false)

u, v, w = model.velocities

#calculating buoyancy from temperature and salinity
#beta = 7.80e-4
#alpha = 1.67e-4
#b = g_Earth * (alpha * model.tracers.T - beta * model.tracers.S)
#b = model.tracers.b

#B = Average(b, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; U, V, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "langmuir_turbulence_averages_$rank.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

run!(simulation)