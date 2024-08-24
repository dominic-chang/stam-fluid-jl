LINEARSOLVERTIMES = 10

function add_source(x, s, dt)
    for I in CartesianIndices(x)
        x[I] += dt * s[I]
    end
end

function set_bnd(M, N, O, b, x)
    # bounds are cells at faces of the cube

    #setting faces
    #x, y cells on z face
    for i in 2:(M-1)
        for j in 2:(N-1)
            x[i, j, 1] = (b == 3 ? -1 : 1)*x[i, j, 2]
            x[i, j, O ] = (b == 3 ? -1 : 1)*x[i, j, O-1]
        end
    end

    #y, z cells on x face
    for i in 2:(N-1)
        for j in 2:(O-1)
            x[1, i, j] = (b == 1 ? -1 : 1)*x[2, i, j]
            x[M, i, j] = (b == 1 ? -1 : 1)*x[M-1, i, j]
        end
    end

    #z, x cells on y face
    for i in 2:(M-1)
        for j in 2:(O-1)
            x[i, 1, j] = (b == 2 ? -1 : 1)*x[i, 2, j]
            x[i, N, j] = (b == 2 ? -1 : 1)*x[i, N-1, j]
        end
    end

    #Setting edges
    # Edges on X face
    for i in 2:(M-1)
        x[i, 1, 1] = 1.0 / 2.0 * (x[i, 2, 1] + x[i, 1, 2])
        x[i, N, 1] = 1.0 / 2.0 * (x[i, N-1, 1] + x[i, N, 2])
        x[i, 1, O] = 1.0 / 2.0 * (x[i, 1, O-1] + x[i, 2, O])
        x[i, N, O] = 1.0 / 2.0 * (x[i, N-1, O] + x[i, N, O-1])
    end

    # Edges on Y face
    for i in 2:(N-1)
        x[1, i, 1] = 1.0 / 2.0 * (x[2, i, 1] + x[1, i, 2])
        x[M, i, 1] = 1.0 / 2.0 * (x[M-1, i, 1] + x[M, i, 2])
        x[1, i, O] = 1.0 / 2.0 * (x[1, i, O-1] + x[2, i, O])
        x[M, i, O] = 1.0 / 2.0 * (x[M-1, i, O] + x[M, i, O-1])
    end

    # Edges on Z face
    for i in 2:(O-1)
        x[1, 1, i] = 1.0 / 2.0 * (x[1, 2, i] + x[2, 1, i  ])
        x[1, N, i] = 1.0 / 2.0 * (x[1, N-1, i] + x[2, N, i  ])
        x[M, 1, i] = 1.0 / 2.0 * (x[M-1, 1, i] + x[M, 2, i  ])
        x[M, N, i] = 1.0 / 2.0 * (x[M, N-1, i] + x[M-1, N, i])
    end

    #setting corners
    x[1, 1, 1] = 1.0 / 3.0 * (x[2, 1, 1] + x[1, 2, 1] + x[1, 1, 2])
    x[1, N, 1] = 1.0 / 3.0 * (x[2, N, 1] + x[1, N-1, 1] + x[1, N, 2])

    x[M, 1, 1] = 1.0 / 3.0 * (x[M, 1, 1] + x[M, 2, 1] + x[M, 1, 2])
    x[M, N, 1] = 1.0 / 3.0 * (x[M-1, N, 1] + x[M, N-1, 1] + x[M, N, 2])

    x[1, 1, O] = 1.0 / 3.0 * (x[2, 1, O] + x[1, 2, O] + x[1, 1, O-1])
    x[1, N, O] = 1.0 / 3.0 * (x[2, N, O] + x[1, N-1, O] + x[1, N, O-1])

    x[M, 1, O] = 1.0 / 3.0 * (x[M-1, 1, O] + x[M, 2, O] + x[M, 1, O-1])
    x[M, N, O] = 1.0 / 3.0 * (x[M-1, N, O] + x[M, N-1, O] + x[M, N, O-1])
end

function linear_solve(M, N, O, b, x, x0, a, c)
    # iterate the solver
    for _ in 1:LINEARSOLVERTIMES
        # update for each cell
        for i in 2:(M-1)
            for j in 2:(N-1)
                for k in 2:(O-1)
                    x[i, j, k] = (x0[i, j, k] + a * (x[i - 1, j, k] + x[i + 1, j, k] + x[i, j - 1,k] + x[i, j + 1, k] + x[i, j, k - 1] + x[i, j, k + 1])) / c
                end
            end
        end
        set_bnd(M, N, O, b, x)
    end
end

function diffuse(M, N, O, b, x, x0, diff, dt)
    mx = max(max(M, N), max(N, O))
    a = dt * diff * mx^3
    linear_solve(M, N, O, b, x, x0, a, 1 + 6 * a)
end

function advect(M, N, O, b, d, d0, u, v, w, dt)

    dtx = dty = dtz = dt * max(max(M, N), max(N, O))

    Threads.@threads for i in 2:(M-1)
        for j in 2:(N-1)
            for k in 2:(O-1)
                x = i - dtx * u[i, j, k]
                y = j - dty * v[i, j, k]
                z = k - dtz * w[i, j, k]
                if (x < 1.5) x = 1.5 end
                if (x > M - 1 + 0.5) x = M-1 + 0.5 end
                i0 = trunc(Int, x)
                i1 = i0 + 1

                if (y < 1.5) y = 1.5 end
                if (y > N - 1 + 0.5) y = N-1 + 0.5 end
                j0 = trunc(Int, y)
                j1 = j0 + 1

                if (z < 1.5) z = 1.5 end
                if (z > O - 1 + 0.5) z = O-1 + 0.5 end
                k0 = trunc(Int, z)
                k1 = k0 + 1

                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1
                u1 = z - k0
                u0 = 1 - u1
                d[i, j, k] = s0 * (t0 * u0 * d0[i0, j0, k0] + t1 * u0 * d0[i0, j1, k0] + t0 * u1 * d0[i0, j0, k1] + t1 * u1 * d0[i0, j1, k1]) +
                                       s1 * (t0 * u0 * d0[i1, j0, k0] + t1 * u0 * d0[i1, j1, k0] + t0 * u1 * d0[i1, j0, k1] + t1 * u1 * d0[i1, j1, k1])
            end
        end
    end
    set_bnd(M, N, O, b, d)
end

function project(M, N, O, u, v, w, p, div)

    for i in 2:(M-1)
        for j in 2:(N-1)
            for k in 2:(O-1)
                div[i, j, k] = -1.0 / 3.0 * ((u[i + 1, j, k] - u[i - 1, j, k]) / M + (v[i, j + 1, k] - v[i, j - 1, k]) / M + (w[i, j, k + 1] - w[i, j, k - 1]) / M)
                p[i, j, k] = 0
            end
        end
    end

    set_bnd(M, N, O, 0, div)
    set_bnd(M, N, O, 0, p)

    linear_solve(M, N, O, 0, p, div, 1, 6)

    for i in 2:(M-1)
        for j in 2:(N-1)
            for k in 2:(O-1)
                u[i, j, k] -= 0.5 * M * (p[i + 1, j, k] - p[i - 1, j, k])
                v[i, j, k] -= 0.5 * M * (p[i, j + 1, k] - p[i, j - 1, k])
                w[i, j, k] -= 0.5 * M * (p[i, j, k + 1] - p[i, j, k - 1])
            end
        end
    end

    set_bnd(M, N, O, 1, u)
    set_bnd(M, N, O, 2, v)
    set_bnd(M, N, O, 3, w)
end

function dens_step(M, N, O, x, x0, u, v, w, diff, dt)
    #add_source(x, x0, dt)
    x .= x0
    x0, x = x, x0
    diffuse(M, N, O, 0, x, x0, diff, dt)
    x0, x = x, x0   
    advect(M, N, O, 0, x, x0, u, v, w, dt)
end

function vel_step(M, N, O, u, v, w, u0, v0, w0, visc, dt)
    #add_source(u, u0, dt)
    #add_source(v, v0, dt)
    #add_source(w, w0, dt)
    u .= u0
    v .= v0
    w .= w0
    u0, u = u, u0
    diffuse(M, N, O, 1, u, u0, visc, dt)
    v0, v = v, v0
    diffuse(M, N, O, 2, v, v0, visc, dt)
    w0, w = w, w0
    diffuse(M, N, O, 3, w, w0, visc, dt)
    project(M, N, O, u, v, w, u0, v0)
    u0, u = u, u0
    v0, v = v, v0
    w0, w = w, w0
    advect(M, N, O, 1, u, u0, u0, v0, w0, dt)
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt)
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt)
    project(M, N, O, u, v, w, u0, v0)
end
