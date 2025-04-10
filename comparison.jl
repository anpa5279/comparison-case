#attempting to match Smitth et al 2016 NCAR LES without angled stokes drift 
using Pkg
using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, hour
using Oceananigans.BuoyancyFormulations: g_Earth

#defining parameters
mutable struct Params
    Nx::Int
    Ny::Int     # number of points in each of horizontal directions
    Nz::Int     # number of points in the vertical direction
    Lx::Float64
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth
end

#defaults, these can be changed directly below
params = Params(8, 8, 8, 128.0, 128.0,1.0)

#global variables
global u₁₀ = 6.0 # (m s⁻¹) wind speed at 10 meters above the ocean

grid = RectilinearGrid(; size=(params.Nx, params.Ny, params.Nz), extent=(params.Lx, params.Ly, params.Lz))
#@show grid
# temperature and salinity boundary conditions
buoyancy = SeawaterBuoyancy()

Q = 1e-6 / hour # [°C s⁻¹]
ρₒ = 1000.0 # [kg m⁻³]
cᴾ = 3985.0 # [J kg⁻¹ K⁻¹]
dTdz = -1e-3 # [°C m⁻¹]

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (ρₒ * cᴾ)),
                                bottom = GradientBoundaryCondition(dTdz))
#@show T_bcs

S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0)) # no salt flux
#@show S_bcs

#functions
function dstokes_dz(z)
    a = 0.1
    b = 5000.0
    Lf = b - a
    nf = 3^9
    df = Lf / nf
    σ = a + 0.5 * df
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀
    for j in 1:length(z)
        u = 0.0
        for i in 1:nf
            u = u + 2.0 * α * g_Earth / (fₚ * σ) * exp(2.0 * σ^2 * z[j] / g_Earth - (fₚ / σ)^4)
            σ = σ + df
        end 
        u0 = df * u #stokes drift 
        u = 0.0
        σ = a + 0.5 * df
        z1 = z[j] + 1e-6
        for i in 1:nf
            u = u + 2.0 * α * g_Earth / (fₚ * σ) * exp(2.0 * σ^2 * z1 / g_Earth - (fₚ / σ)^4)
            σ = σ + df
        end 
        u1 = df * u #stokes drift
        dudz[j] = (u1 - u0) / (1e-6)
        @show z[j]
        @show u
        @show dudz[j]
    end
    return dudz
end 
@show dstokes_dz
@show grid.Nz
dudz = Array{Float64}(undef, grid.Nz)
dudz = dstokes_dz(znodes(grid, Center()))
@inline ∂z_uˢ(z, t) =

τx = 0.025 # N m⁻²
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
#@show u_bcs

#coriolis = FPlane(f=0.729e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))
#@show model

# Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = 20 + dTdz * z 

u★ = sqrt(abs(τx))
uᵢ(x, y, z) = u★ 
wᵢ(x, y, z) = u★ 

set!(model, u=uᵢ, w=wᵢ, T=Tᵢ, S=35)

simulation = Simulation(model, Δt=45.0, stop_time = 4hours)
#@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "comparison_fields.jld2",
                                                      overwrite_existing = true,
                                                      with_halos = false)

u, v, w = model.velocities

U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; U, V, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "comparison_averages.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

run!(simulation)