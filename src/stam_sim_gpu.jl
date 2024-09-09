# Julia port of Jos Stam's fluid solver
# https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

using Pkg;Pkg.activate(dirname(@__DIR__))
using WGLMakie
import Metal
WGLMakie.activate!() # hide
WGLMakie.Makie.inline!(true) # hide
include("stam_solver_gpu.jl")

SIZE = 512

M = SIZE; # grid x
N = SIZE; # grid y
O = 4; # grid z
dt = 0.4f0; # time delta
diff = 100f0; # diffuse
visc = 0f0; # viscosity
force = 10f0;  # added on keypress on an axis
source = 200f0; # density
source_alpha = 0.05f0; #for displaying density

addforce = Observable([0, 0, 0])
addsource = Observable(0);
vx = Observable(Metal.zeros(Float32, M, N))
vy = Observable(Metal.zeros(Float32, M, N))
vx_prev = Observable(Metal.zeros(Float32, M, N))
vy_prev = Observable(Metal.zeros(Float32, M, N))
dens = Observable(Metal.zeros(Float32, M, N))
dens_prev = Observable(Metal.MtlMatrix(map(I -> Float32(exp(-hypot((Tuple(I) .- (M / 2, N / 2))...)^2 / 2)/4), CartesianIndices(dens[]))))

backend = get_backend(dens[])
function sim_main()
    #get_force_source( dens_prev, vx_prev, vy_prev, vz_prev );
    vel_step!(vx[], vy[], vx_prev[], vy_prev[], visc, dt)
    dens_step!(dens[], dens_prev[], vx[], vy[], diff, dt)
end
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
on(events(fig).tick) do tick
    sim_main()
    notify(dens)
    println(events(fig).tick)

    plot!(ax, Array(dens[])[:,:,1])
end

#set_bnd!(backend)(dens_prev[], 0, ndrange = (M, N))
#vel_step!(vx[], vy[], vx_prev[], vy_prev[], visc, dt)