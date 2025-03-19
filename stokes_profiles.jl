using Pkg
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, hour
using Oceananigans.BuoyancyFormulations: g_Earth
using CairoMakie
using Printf

#functions
function stokes_kernel(f, z, u10=6.0)
    α = 0.00615
    g = 9.81
    fₚ = 2π * 0.13 * g / u10
    return 2.0 * α * g / (fₚ * f) * exp(2.0 * f^2 * z / g - (fₚ / f)^4)
end

function stokes_velocity(z)
    a = 0.1
    b = 5000.0
    Lf = b - a
    nf = 3^9
    df = Lf / nf
    σ = a + 0.5 * df
    u = 0.0
    for _ in 1:nf
        u = u + stokes_kernel(σ, z)
        σ = σ + df
    end 
    return df * u
end

function dstokes_dz(z, t)
    u0 = stokes_velocity(z)
    u1 = stokes_velocity(z + 1e-6)
    dudz = (u1 - u0) / (1e-6)
    return dudz
end 

function plot()
    #grid setup
    grid = RectilinearGrid(size = 128, z = (-96.0, 0.0), topology=(Flat, Flat, Bounded))
    Nz = grid.Nz
    Lz = grid.Lz
    z = grid.z.cᵃᵃᶜ[1:Nz]
    u_data = Array{Float64}(undef, (1, 1, grid.Nz))
    dudz_data = Array{Float64}(undef, (1, 1, grid.Nz))

    #calculating stokes drift in z-direction
    for k in 1:Nz
        z_pt = z[k]
        u_data[k] = stokes_velocity(z_pt)
        dudz_data[k] = dstokes_dz(z_pt, 0.0)
    end

    u = FieldTimeSeries{Face, Center, Center}(grid, 0.0)
    dudz = FieldTimeSeries{Face, Center, Center}(grid, 0.0)

    u .= u_data
    dudz .= dudz_data

    #plotting velocity profile
    n = Observable(1)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "uˢ [m/s]", ylabel = "z [m]", title = "Stokes drift")
    Uₙ = @lift view(u[$n], 1, 1, :)
    lines!(ax, Uₙ)
    fig 
    save("stokes_drift.png", fig)
    #plotting velocity gradient profile
    fig2 = Figure()
    ax2 = Axis(fig2[1, 1], xlabel = "duˢ/dz [1/s]", ylabel = "z [m]", title = "Stokes drift gradient")
    dUₙ = @lift view(dudz[$n], 1, 1, :)
    lines!(ax2, dUₙ)
    fig2
    save("stokes_drift_gradient.png", fig2)
end 
