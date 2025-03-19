#attempting to match Smitth et al 2016 NCAR LES without angled stokes drift 
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
end

#defaults, these can be changed directly below
params = Params(32, 32, 32, 128.0, 128.0,160.0)

#functions
function stokes_kernel(f, z, u10)
    α = 0.00615
    g = 9.81
    fₚ = 2π * 0.13 * g / u10
    return 2.0 * α * g / (fₚ * f) * exp(2.0 * f^2 * z / g - (fₚ / f)^4)
end

function stokes_velocity(z, u10)
    a = 0.1
    b = 5000.0
    Lf = b - a
    nf = 3^9
    df = Lf / nf
    σ = a + 0.5 * df
    u = 0.0
    for _ in 1:nf
        u = u + stokes_kernel(σ, z, u10)
        σ = σ + df
    end 
    return df * u
end

# Automatically distributing among available processors
arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), extent=(params.Lx, params.Ly, params.Lz))
@show grid

# temperature and salinity boundary conditions
buoyancy = SeawaterBuoyancy()

Q = 1e-6 / hour # [°C s⁻¹]
ρₒ = 1000.0 # [kg m⁻³]
cᴾ = 3985.0 # [J kg⁻¹ K⁻¹]
dTdz = -1e-3 # [°C m⁻¹]

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (ρₒ * cᴾ)),
                                bottom = GradientBoundaryCondition(dTdz))

@show T_bcs

@inline Jˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹

const evaporation_rate = 1e-3 / hour # m s⁻¹

evaporation_bc = FluxBoundaryCondition(Jˢ, field_dependencies=:S, parameters= evaporation_rate)

S_bcs = FieldBoundaryConditions(top=evaporation_bc)
@show S_bcs

# Stokes drift profile
u₁₀ = 10    # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
τx = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻², surface stress

uˢ(z) = stokes_velocity(z, u₁₀)

∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

τx = 0.025 # N m⁻²
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@show u_bcs

coriolis = FPlane(f=0.729e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))
@show model

@inline Ξ(z) = randn() * exp(z / 4)

# Temperature initial condition: a stable density gradient with random noise superposed.
@inline Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

u★ = sqrt(abs(τx))
@inline uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
@inline wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, T=Tᵢ, S=35)

simulation = Simulation(model, Δt=45.0, stop_time = 4hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "camparison_fields$rank.jld2",
                                                      overwrite_existing = true,
                                                      with_halos = false)

u, v, w = model.velocities

U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; U, V, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "comparison_averages_$rank.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

run!(simulation)