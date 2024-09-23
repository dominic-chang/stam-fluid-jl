import Metal
using KernelAbstractions

@kernel function _add_source!(x, s, dt)
    I = @index(Global)
    @inbounds x[I] += dt*s[I]
end

function add_source!(x, s, dt)
    backend = get_backend(x)
    _add_source!(backend)(x, s, dt, ndrange = length(x)) 
end

@kernel function set_bnd!(x, b)
    # This is the continuity boundary condition from the paper
    I, J = @index(Global, NTuple)
    T = typeof(I)
    ans = x[I,J]
    M, N = T.(size(x))
    if I == 1
        if J == 1
            ans = (x[2,1] + x[1,2])/2
        elseif J == N
            ans = (x[2,N] + x[1,N-1])/2
        else
            ans = b == 1 ? -x[2,J] : x[2,J]
        end
    elseif I == M
        if J == 1
            ans = (x[M-1,1] + x[M,2])/2
        elseif J == N
            ans = (x[M-1,N] + x[M,N-1])/2
        else
            ans = b == 1 ? -x[M-1,J] : x[M-1,J]
        end
    elseif J == 1
        ans = b == 2 ? -x[I,2] : x[I,2]
    elseif J == N
        ans = b == 2 ? -x[I,N-1] : x[I,N-1]
    end
    x[I, J] = ans   
end

@kernel function _linear_solve!(x, x0, a, c)
    I,J = @index(Global, NTuple)
    T = typeof(I)
    M, N = T.(size(x))
    if (1 < I < M) && (1 < J < N)
        @inbounds x[I, J] = (x0[I, J] + a*(x[I-1, J] + x[I+1, J] + x[I, J-1] + x[I, J+1])) / c
    end
end

function linear_solve!(x, x0, a, b, c; linearsolvertimes=10)
    backend = get_backend(x)
    @assert size(x) == size(x0)
    @assert get_backend(x0) == backend

    M, N = size(x)
    for _ in 1:linearsolvertimes
        _linear_solve!(backend)(x, x0, a, c, ndrange = (M, N))
        set_bnd!(backend)(x, b, ndrange = (M, N))
    end
end

# Diffuses field x over time dt
function diffuse!(d, d0, b, diff, dt)
    mx = max(size(d)...)
    a = dt * diff * mx^2
    linear_solve!(d, d0, a, b, 1 + 4 * a)
end

# Advects field d along field (vx, vy) over time dt
@kernel function _advect!(d, d0, vx, vy, dt::A) where A
    I, J = @index(Global, NTuple)
    T = typeof(I)
    M, N = T.(size(d))
    dtx = dty = dt * A(max(M, N))
    if (1 < I < M) && (1 < J < N)
        x = A(I) - dtx * vx[I, J]
        y = A(J) - dty * vy[I, J]
        if x < A(1/2)
            x = A(1/2)
        end
        if x > (A(M) + A(1/2))
            x = A(M) + A(1/2)
        end
        i0 = floor(x)
        i1 = i0 + 1
        if y < A(1/2)
            y = A(1/2)
        end
        if y > (A(N) + A(1/2))
            y = A(N) + A(1/2)
        end
        j0 = floor(y)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1

        i0 = unsafe_trunc(T, i0)
        i1 = unsafe_trunc(T, i1)
        j0 = unsafe_trunc(T, j0)
        j1 = unsafe_trunc(T, j1)
        d[I, J] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1] )
    end
end
function advect!(d, d0, vx, vy, dt)
    backend = get_backend(d)
    @assert size(d) == size(d0)
    @assert get_backend(d0) == backend

    M, N = size(d)
    _advect!(backend)(d, d0, vx, vy, dt, ndrange = (M, N))
    set_bnd!(backend)(d, 0, ndrange = (M, N))
end

@kernel function _project_div_p(u, v, p, div)
    I, J = @index(Global, NTuple)
    T = typeof(I)
    M, N = T.(size(u))
    h = max(M, N)
    if (1 < I < M) && (1 < J < N)
        div[I, J] = -(u[I+1, J] - u[I-1, J] + v[I, J+1] - v[I, J-1]) / (2*h)
        p[I, J] = 0
    end
end

@kernel function _project2(u, v, p)
    I, J = @index(Global, NTuple)
    T = typeof(I)
    M,N = T.(size(u))
    h = max(M, N)
    if (1 < I < M) && (1 < J < N)
        u[I, J] -= (p[I+1, J] - p[I-1, J])/ (2 * h )
        v[I, J] -= (p[I, J+1] - p[I, J-1])/ (2 * h )
    end
end

function project(u, v, p, div)
    backend = get_backend(u)
    @assert size(u) == size(v) == size(p) == size(div)
    @assert get_backend(v) == get_backend(p) == get_backend(div) == backend

    M, N = size(u)
    _project_div_p(backend)(u, v, p, div, ndrange = (M, N))
    set_bnd!(backend)(div, 0, ndrange = (M, N))
    set_bnd!(backend)(p, 0, ndrange = (M, N))
    linear_solve!(p, div, 1, 0, 4, linearsolvertimes=10)
    _project2(backend)(u, v, p, ndrange = (M, N))
    set_bnd!(backend)(u, 1, ndrange = (M, N))
    set_bnd!(backend)(v, 2, ndrange = (M, N))
end

function dens_step!(x, x0, u, v, diff, dt)
    #add_source!(x, x0, dt)
    mycopy!(x, x0)
    #x, x0 = x0, x
    diffuse!(x, x0, 0, diff, dt)
    x, x0 = x0, x
    advect!(x, x0, u, v, dt)
end

function vel_step!(u, v, u0, v0, visc, dt)
    backend = get_backend(u)
    copy_kernel!(backend)(u, u0, ndrange = size(u))
    copy_kernel!(backend)(v, v0, ndrange = size(v))
    u, u0 = u0, u
    v, v0 = v0, v
    diffuse!(u, u0, 1, visc, dt)
    diffuse!(v, v0, 2, visc, dt)
    project(u, v, u0, v0)
    u, u0 = u0, u
    v, v0 = v0, v
    advect!(u, u0, u0, v0, dt)
    advect!(v, v0, u0, v0, dt)
    project(u, v, u0, v0)
end

@kernel function copy_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

function mycopy!(A, B)
    backend = get_backend(A)
    @assert size(A) == size(B)
    @assert get_backend(B) == backend

    kernel = copy_kernel!(backend)
    kernel(A, B, ndrange = length(A))
end

backend = Metal.MetalBackend()

A = KernelAbstractions.zeros(backend, Float32, 128, 128)
B = KernelAbstractions.ones(backend, Float32, 128, 128)
mycopy!(A, B)
KernelAbstractions.synchronize(backend)


