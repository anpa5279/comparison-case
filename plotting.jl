using Pkg
using Statistics
using CairoMakie
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using Oceananigans.BuoyancyFormulations: g_Earth

function plot()
    # running locally: using Pkg; Pkg.add("Oceananigans"); Pkg.add("CairoMakie"); Pkg.add("JLD2")
    Nranks = 4

    fld_file="outputs/comparison_fields_0.jld2"
    averages_file="outputs/comparison_averages_0.jld2"

    w_temp = FieldTimeSeries(fld_file, "w")
    u_temp = FieldTimeSeries(fld_file, "u")
    T_temp = FieldTimeSeries(fld_file, "T")
    S_temp = FieldTimeSeries(fld_file, "S")
    U_temp = FieldTimeSeries(averages_file, "U")
    V_temp = FieldTimeSeries(averages_file, "V")
    wu_temp = FieldTimeSeries(averages_file, "wu")
    wv_temp = FieldTimeSeries(averages_file, "wv")

    Lx = Nranks * u_temp.grid.Lx
    Ly = u_temp.grid.Ly
    Lz = u_temp.grid.Lz
    Nx = Nranks * u_temp.grid.Nx
    Ny = u_temp.grid.Ny
    Nz = u_temp.grid.Nz
    Nt = length(u_temp.times)
    grid = RectilinearGrid(size = (Nx, Ny, Nz), extent = (Lx, Ly, Lz))
    times = u_temp.times

    w_data = Array{Float64}(undef, (Nx, Ny, Nz + 1, Nt)) #because face value
    u_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    T_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    S_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    U_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    V_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    wu_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))  
    wv_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))
    U_data .= 0
    V_data .= 0
    wu_data .= 0
    wv_data .= 0

    p = 1
    w_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= w_temp.data
    u_data[p:p + u_temp.grid.Nx - 1, :, :, :] .= u_temp.data
    T_data[p:p + T_temp.grid.Nx - 1, :, :, :] .= T_temp.data
    S_data[p:p + S_temp.grid.Nx - 1, :, :, :] .= S_temp.data
    U_data .= U_data .+ U_temp.data
    V_data .= V_data .+ V_temp.data
    wu_data .= wu_data .+ wu_temp.data
    wv_data .= wv_data .+ wv_temp.data

    for i in 1:Nranks-1

        p = p + u_temp.grid.Nx

        println("Loading rank $i")

        fld_file="outputs/comparison_fields_rank$(i).jld2"
        averages_file="outputs/comparison_averages_rank$(i).jld2"

        w_temp = FieldTimeSeries(fld_file, "w")
        u_temp = FieldTimeSeries(fld_file, "u")
        T_temp = FieldTimeSeries(fld_file, "T")
        S_temp = FieldTimeSeries(fld_file, "S")
        U_temp = FieldTimeSeries(averages_file, "U")
        V_temp = FieldTimeSeries(averages_file, "V")
        wu_temp = FieldTimeSeries(averages_file, "wu")
        wv_temp = FieldTimeSeries(averages_file, "wv")
        
        w_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= w_temp.data
        u_data[p:p + u_temp.grid.Nx - 1, :, :, :] .= u_temp.data
        T_data[p:p + T_temp.grid.Nx - 1, :, :, :] .= T_temp.data
        S_data[p:p + S_temp.grid.Nx - 1, :, :, :] .= S_temp.data
        U_data .= U_data .+ U_temp.data
        V_data .= V_data .+ V_temp.data
        wu_data .= wu_data .+ wu_temp.data
        wv_data .= wv_data .+ wv_temp.data
        
    end

    #averaging
    U_data = U_data ./ Nranks
    V_data = V_data ./ Nranks
    wu_data = wu_data ./ Nranks
    wv_data = wv_data ./ Nranks

    #calculating buoyancy from temperature and salinity
    beta = 7.80e-4
    alpha = 1.67e-4
    B_temp = Array{Float64}(undef, (1, 1, Nz, Nt))
    B_temp = g_Earth * (alpha * T_data - beta * S_data)
    B_data = Statistics.mean(B_temp, dims=(1, 2))
    
    #putting everything back into FieldTimeSeries
    w = FieldTimeSeries{Center, Center, Face}(grid, times)
    u = FieldTimeSeries{Face, Center, Center}(grid, times)
    T = FieldTimeSeries{Face, Center, Center}(grid, times)
    S = FieldTimeSeries{Face, Center, Center}(grid, times)
    B = FieldTimeSeries{Center, Center, Center}(grid, times)
    U = FieldTimeSeries{Center, Center, Center}(grid, times)
    V = FieldTimeSeries{Center, Center, Center}(grid, times)
    wu = FieldTimeSeries{Center, Center, Face}(grid, times)
    wv = FieldTimeSeries{Center, Center, Face}(grid, times)

    w .= w_data
    u .= u_data
    T .= T_data
    S .= S_data
    B .= B_data
    U .= U_data
    V .= V_data
    wu .= wu_data
    wv .= wv_data

    @show w

    #begin plotting
    n = Observable(1)

    wxy_title = @lift string("w(x, y, t) at z=-8 m and t = ", prettytime(times[$n]))
    wxz_title = @lift string("w(x, z, t) at y=0 m and t = ", prettytime(times[$n]))
    uxz_title = @lift string("u(x, z, t) at y=0 m and t = ", prettytime(times[$n]))

    xT, yT, zT = nodes(T)

    axis_kwargs = (xlabel="x (m)",
                ylabel="z (m)",
                aspect = AxisAspect(Lx/Lz),
                limits = ((0, Lx), (-Lz, 0)))

    fig = Figure(size = (850, 1150))

    ax_B = Axis(fig[1, 4:5];
                xlabel = "Buoyancy (m s⁻²)", xtickformat = "{:.3f}", xticklabelrotation = pi/4,
                ylabel = "z (m)")
    
    ax_U = Axis(fig[2, 4:5];
                xlabel = "Velocities (m s⁻¹)",
                ylabel = "z (m)",
                limits = ((-0.07, 0.07), nothing))

    ax_fluxes = Axis(fig[3, 4:5];
                    xlabel = "Momentum fluxes (m² s⁻²)", xticklabelrotation = pi/4,
                    ylabel = "z (m)",
                    limits = ((-3.5e-5, 3.5e-5), nothing))

    ax_wxy = Axis(fig[1, 1:2];
                xlabel = "x (m)",
                ylabel = "y (m)",
                aspect = DataAspect(),
                limits = ((0, Lx), (0, Ly)),
                title = wxy_title)

    ax_wxz = Axis(fig[2, 1:2]; title = wxz_title, axis_kwargs...)

    ax_uxz = Axis(fig[3, 1:2]; title = uxz_title, axis_kwargs...)

    ax_T  = Axis(fig[4, 1:2]; title = "Temperature", axis_kwargs...)

    ax_S  = Axis(fig[4, 4:5]; title = "Salinity", axis_kwargs...)
    title = @lift @sprintf("t = %s", prettytime(times[$n]))

    wₙ = @lift w[$n]
    uₙ = @lift u[$n]
    Tₙ = @lift interior(T[$n],  :, 1, :)
    Sₙ = @lift interior(S[$n],  :, 1, :)
    Bₙ = @lift view(B[$n], 1, 1, :)
    Uₙ = @lift view(U[$n], 1, 1, :)
    Vₙ = @lift view(V[$n], 1, 1, :)
    wuₙ = @lift view(wu[$n], 1, 1, :)
    wvₙ = @lift view(wv[$n], 1, 1, :)

    k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
    wxyₙ = @lift view(w[$n], :, :, k)
    wxzₙ = @lift view(w[$n], :, 1, :)
    uxzₙ = @lift view(u[$n], :, 1, :)

    wlims = (-0.03, 0.03)
    ulims = (-0.05, 0.05)
    Tlims = (19.7, 19.99)
    Slims = (35, 35.005)

    lines!(ax_B, Bₙ)

    lines!(ax_U, Uₙ; label = L"\bar{u}")
    lines!(ax_U, Vₙ; label = L"\bar{v}")
    axislegend(ax_U; position = :rb)

    lines!(ax_fluxes, wuₙ; label = L"mean $wu$")
    lines!(ax_fluxes, wvₙ; label = L"mean $wv$")
    axislegend(ax_fluxes; position = :rb)

    hm_T = heatmap!(ax_T, xT, zT, Tₙ; colormap = :thermal, colorrange = Tlims)
    Colorbar(fig[4, 3], hm_T; label = "ᵒC")

    hm_wxy = heatmap!(ax_wxy, wxyₙ;
                    colorrange = wlims,
                    colormap = :balance)

    Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

    hm_wxz = heatmap!(ax_wxz, wxzₙ;
                    colorrange = wlims,
                    colormap = :balance)

    Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

    ax_uxz = heatmap!(ax_uxz, uxzₙ;
                    colorrange = ulims,
                    colormap = :balance)

    Colorbar(fig[3, 3], ax_uxz; label = "m s⁻¹")
    
    hm_S = heatmap!(ax_S, xT, zT, Sₙ; colormap = :haline, colorrange = Slims)
    Colorbar(fig[4, 6], hm_S; label = "g / kg")

    fig

    frames = 1:length(times)

    record(fig, "comparison_case.mp4", frames, framerate=8) do i
        n[] = i
    end

end
