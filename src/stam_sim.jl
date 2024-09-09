# Julia port of Jos Stam's fluid solver
# https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

using Pkg;Pkg.activate(dirname(@__DIR__))
using WGLMakie
WGLMakie.activate!() # hide
WGLMakie.Makie.inline!(true) # hide
include("stam_solver.jl")

SIZE = 100

M = SIZE; # grid x
N = SIZE; # grid y
O = 4; # grid z
dt = 0.4; # time delta
diff = 100.0; # diffuse
visc = 0.0; # viscosity
force = 10.0;  # added on keypress on an axis
source = 200.0; # density
source_alpha = 0.05; #for displaying density

addforce = Observable([0, 0, 0])
addsource = Observable(0);
vx = Observable(zeros(Float64, M, N, O))
vy = Observable(zeros(Float64, M, N, O))
vz = Observable(zeros(Float64, M, N, O))
vx_prev = Observable(zeros(Float64, M, N, O))
vy_prev = Observable(zeros(Float64, M, N, O))
vz_prev = Observable(zeros(Float64, M, N, O))
dens = Observable(zeros(Float64, M, N, O))
dens_prev = Observable(zeros(Float64, M, N, O))
map(I -> dens_prev[][I] = exp(-hypot((Tuple(I) .- (M / 2, N / 2, O / 2))...)^2 / 2)/4, CartesianIndices(dens[]))

dvel = 0;
dhelp = 1;
daxis = 1;

function clear_data()
    size = M * N * O

    for i in 1:size
        vx[i] = vy[i] = vz[i] = vx_prev[i] = vy_prev[i] = vz_prev[i] = dens[i] = dens_prev[i] = 0.0
    end

    addforce[1] = addforce[2] = addforce[3] = 0
end


function get_force_source(d, u, v, w)
    global addsource
    for I in CartesianIndices(u)
        vx[I] = vy[I] = vz[I] = d[I] = 0.0
    end

    if (addforce[1] == 1) # x
        i = 3
        j = N / 2
        k = O / 2

        if (i ≤ 1 || i ≥ M || j ≤ 1 || j ≥ N || k ≤ 1 || k ≥ O)
            return
        end
        vy[i, j, k] = force * 10
        addforce[2] = 0

    end

    if (addforce[2] == 1)
        i = M / 2
        j = 3
        k = O / 2

        if (i ≤ 1 || i ≥ M || j ≤ 1 || j ≥ N || k ≤ 1 || k ≥ O)
            return
        end
        vy[i, j, k] = force * 10
        addforce[2] = 0
    end

    if (addforce[3] == 1) # y
        i = M / 2
        j = N / 2
        k = 3

        if (i ≤ 1 || i ≥ M || j ≤ 1 || j ≥ N || k ≤ 1 || k ≥ O)
            return
        end
        vy[i, j, k] = force * 10
        addforce[3] = 0

    end

    if (addsource == 1)
        i = M / 2
        j = N / 2
        k = O / 2
        d[i, j, k] = source
        addsource = 0
    end

    return
end
function sim_main()
    #get_force_source( dens_prev, vx_prev, vy_prev, vz_prev );
    #vel_step(vx[], vy[], vz[], vx_prev[], vy_prev[], vz_prev[], visc, dt)
    dens_step(dens[], dens_prev[], vx[], vy[], vz[], diff, dt)
end
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
fig
on(events(fig).tick) do tick
    sim_main()
    notify(dens)
    println(maximum(dens[]))

    plot!(ax, dens_prev[][:,:,1])
end

#points = Observable(Point2f[])
#
#scene = Scene(camera = campixel!)
#linesegments!(scene, points, color = :black)
#scatter!(scene, points, color = :gray)
#
#on(events(scene).mousebutton) do event
#    if event.button == Mouse.left
#        if event.action == Mouse.press || event.action == Mouse.release
#            mp = events(scene).mouseposition[]
#            push!(points[], mp)
#            notify(points)
#        end
#    end
#end
#
#scene