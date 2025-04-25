using Pkg
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, hour
using Oceananigans.BuoyancyFormulations: g_Earth
using CairoMakie
using DelimitedFiles
using Printf

#functions
#stokes drift
function stokes_kernel(f, z, u₁₀)
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    return 2.0 * α * g_Earth / (fₚ * f) * exp(2.0 * f^2 * z / g_Earth - (fₚ / f)^4)
end
function stokes_velocity(z, u₁₀)
    u = Array{Float64}(undef, length(z))
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ1 = a + 0.5 * df
        u_temp = 0.0
        for k in 1:nf
            u_temp = u_temp + stokes_kernel(σ1, z[i], u₁₀)
            σ1 = σ1 + df
        end 
        u[i] = df * u_temp
    end
    return u
end
function dstokes_dz(z, u₁₀)
    dudz = Array{Float64}(undef, length(z))
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ = a + 0.5 * df
        du_temp = 0.0
        for k in 1:nf
            du_temp = du_temp + (4.0 * α * σ/ (fₚ) * exp(2.0 * σ^2 * z[i] / g_Earth - (fₚ / σ)^4))
            σ = σ + df
        end 
        dudz[i] = df * du_temp
    end
    return dudz
end 
#calculating stokes drift in z-direction
z = collect(znodes(grid, Center()))
u_data = Array{Float64}(undef, (Nz))
dudz_data = Array{Float64}(undef, (Nz))
for k in 1:Nz
    z_pt = z[k]
    u_data[k] = stokes_velocity(z_pt, 5.75)[1]
    dudz_data[k] = dstokes_dz(z_pt, 5.75)[1]
end
#read in ncar les profiles
ncarles = readdlm("NCAR-LES-stokes.txt", ' ')
ncarles_stokes = reverse(ncarles[:, 3])
ncarles_z = reverse(ncarles[:, 2])
ncarles_dudz = Array{Float64}(undef, length(ncarles_z)-1)
for i in 1:length(ncarles_z) - 1
    ncarles_dudz[i] = (ncarles_stokes[i] - ncarles_stokes[i + 1])/(ncarles_z[i] - ncarles_z[i + 1])
end
#plotting velocity profile
n = Observable(1)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "uˢ [m/s]", ylabel = "z [m]", title = "Stokes drift")
lines!(ax, u_data, z, label = "Oceananigans Stokes drift")
lines!(ax, ncarles_stokes, ncarles_z, label = "NCAR-LES Stokes drift")
axislegend(ax; position = :rb)
fig 
save("stokes_drift.png", fig)
#plotting velocity gradient profile
fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel = "duˢ/dz [1/s]", ylabel = "z [m]", title = "Stokes drift gradient")
lines!(ax2, dudz_data, z, label = "Oceananigans Stokes drift gradient")
lines!(ax2, ncarles_dudz, ncarles_z[2:length(ncarles_z)], label = "NCAR-LES Stokes drift gradient")
fig2
axislegend(ax2; position = :rb)
save("stokes_drift_gradient.png", fig2)