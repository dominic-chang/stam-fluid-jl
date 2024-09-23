# Julia port of Jos Stam's fluid solver
# https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

using Pkg;Pkg.activate(dirname(@__DIR__))
using WGLMakie
import Metal
WGLMakie.activate!() # hide
WGLMakie.Makie.inline!(true) # hide
include("stam_solver_gpu.jl")

SIZE = 128

M = SIZE; # grid x
N = SIZE; # grid y
O = 4; # grid z
dt = 0.1f0; # time delta
diff = 0.00001f0; # diffuse
visc = 0f0; # viscosity
force = 10f0;  # added on keypress on an axis
source = 200f0; # density
source_alpha = 0.05f0; #for displaying density

addforce = Observable([0, 0, 0])
addsource = Observable(0);
vx = Observable(0.01f0Metal.ones(Float32, M, N))
vy = Observable(Metal.zeros(Float32, M, N))
vx_prev = Observable(Metal.rand(Float32, M, N))
vy_prev = Observable(0.01f0Metal.ones(Float32, M, N))
dens = Observable(Metal.zeros(Float32, M, N))
dens_prev = Observable(Metal.zeros(Float32, M, N))#Metal.MtlMatrix(map(I -> Float32(exp(-hypot((Tuple(I) .- (M / 2, N / 2))...)^2 / 2)/4), CartesianIndices(dens[]))))

backend = get_backend(dens[])
function sim_main()
    #get_force_source( dens_prev, vx_prev, vy_prev, vz_prev );
    #vel_step!(vx[], vy[], vx_prev[], vy_prev[], visc, dt)
    dens_step!(dens[], dens_prev[], vx[], vy[], diff, dt)
end
fig = Figure(resolution = (800, 800))
ax = Axis(fig[1, 1], aspect=1)
ax1 = Axis(fig[1, 2], aspect=1)
hidedecorations!(ax)
on(events(fig).tick) do tick
    sim_main()
    notify(dens)
    #println(events(fig).tick)
    if tick.count % 5 == 0
        plot!(ax, Array(dens[])[:,:,1], colorrange = (0, 0.1), colormap = :thermal)
        plot!(ax1, Array(dens_prev[])[:,:,1], colorrange = (0, 0.1), colormap = :thermal)
    end
end
register_interaction!(ax, :my_interaction) do event::MouseEvent, axis
    if event.type === MouseEventTypes.leftclick
        println("You clicked on the axis at datapos $(event.data)")
        pos = unsafe_trunc.(Ref(Int), event.data)
        Metal.GPUArrays.@allowscalar dens_prev[][pos...]  += 1.0
    end
end