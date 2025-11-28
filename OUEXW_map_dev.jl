

# Trilinear interpolation in unit cube
function trilinear_normalized(xd::Real, yd::Real, zd::Real, vals::Array{Float64, 3})
    v000 = vals[1, 1, 1]
    v100 = vals[2, 1, 1]
    v010 = vals[1, 2, 1]
    v110 = vals[2, 2, 1]
    v001 = vals[1, 1, 2]
    v101 = vals[2, 1, 2]
    v011 = vals[1, 2, 2]
    v111 = vals[2, 2, 2]

    return (v000 * (1 - xd) * (1 - yd) * (1 - zd) +
            v100 * xd * (1 - yd) * (1 - zd) +
            v010 * (1 - xd) * yd * (1 - zd) +
            v110 * xd * yd * (1 - zd) +
            v001 * (1 - xd) * (1 - yd) * zd +
            v101 * xd * (1 - yd) * zd +
            v011 * (1 - xd) * yd * zd +
            v111 * xd * yd * zd)
end

# Bilinear interpolation in unit square
function bilinear_normalized(u::Real, v::Real, vals::Matrix{Float64})
    return ((1 - u) * (1 - v) * vals[1, 1] +
            u * (1 - v) * vals[2, 1] +
            (1 - u) * v * vals[1, 2] +
            u * v * vals[2, 2])
end

# transform SLIF parameters (μₛ, λₛ, τₛ) to Exwald parameters (μₓ, λₓ, τₓ)
# By trilinear interpolation in 3d grid given by
# EXWparam, (mu_o, lambda_o,tau_o) computed by map_OU2Exwald()
function SLIF2Exwald(OUparam::Tuple{Float64, Float64, Float64},
                    grid::Tuple{Vector{Float64},Vector{Float64}, Vector{Float64}},
                    O2X_map::Array{Tuple{Float64, Float64, Float64}, 3})
    
    mu_o, lambda_o, tau_o = OUparam
    muo_grid, lambdao_grid, tauo_grid = grid

    n = length(muo_grid)
    if n != size(O2X_map, 1) || n != size(O2X_map, 2) || n != size(O2X_map, 3)
        throw(ArgumentError("Grids must have length equal to O2X dimensions (40)"))
    end

    # Find voxel indices (1-based)
    i = searchsortedfirst(muo_grid, mu_o) - 1
    j = searchsortedfirst(lambdao_grid, lambda_o) - 1
    k = searchsortedfirst(tauo_grid, tau_o) - 1

    if i < 1 || i >= n || muo_grid[i] > mu_o || muo_grid[i+1] < mu_o ||
       j < 1 || j >= n || lambdao_grid[j] > lambda_o || lambdao_grid[j+1] < lambda_o ||
       k < 1 || k >= n || tauo_grid[k] > tau_o || tauo_grid[k+1] < tau_o
        return (NaN, NaN, NaN)
    end

    # Check all 8 vertices valid
    valid = true
    for ii in 0:1, jj in 0:1, kk in 0:1
        if isnan(O2X_map[i + ii, j + jj, k + kk][1])
            valid = false
            break
        end
    end
    if !valid
        return (NaN, NaN, NaN)
    end


    # Normalized coordinates
    xd = (mu_o - muo_grid[i]) / (muo_grid[i + 1] - muo_grid[i])
    yd = (lambda_o - lambdao_grid[j]) / (lambdao_grid[j + 1] - lambdao_grid[j])
    zd = (tau_o - tauo_grid[k]) / (tauo_grid[k + 1] - tauo_grid[k])

    # Build vals for each component
    mu_vals = zeros(2, 2, 2)
    lambda_vals = zeros(2, 2, 2)
    tau_vals = zeros(2, 2, 2)
    for ii in 0:1, jj in 0:1, kk in 0:1
        t = O2X_map[i + ii, j + jj, k + kk]
        mu_vals[ii + 1, jj + 1, kk + 1] = t[1]
        lambda_vals[ii + 1, jj + 1, kk + 1] = t[2]
        tau_vals[ii + 1, jj + 1, kk + 1] = t[3]
    end

    mux = trilinear_normalized(xd, yd, zd, mu_vals)
    lambdax = trilinear_normalized(xd, yd, zd, lambda_vals)
    taux = trilinear_normalized(xd, yd, zd, tau_vals)

    return (mux, lambdax, taux)
end

# transform Exwald parameters (μₓ, λₓ, τₓ) to  SLIF parameters (μₛ, λₛ, τₛ)
# By Newton-Raphson search in 3d grid given by
# O2X_map, grid = (mu_o, lambda_o,tau_o) computed by map_OU2Exwald()
function Exwald2SLIF(EXWparam::Tuple{Float64, Float64, Float64},
                    grid::Tuple{Vector{Float64}, Vector{Float64},Vector{Float64}},
                    O2X_map::Array{Tuple{Float64, Float64, Float64}, 3})
    
    mu_x, lambda_x, tau_x = EXWparam
    muo_grid, lambdao_grid, tauo_grid = grid

    n = length(muo_grid)
    if n != size(O2X_map, 1) || n != size(O2X_map, 2) || n != size(O2X_map, 3)
        throw(ArgumentError("Grids must have length equal to O2X dimensions (40)"))
    end

    # Compute global bounds from valid points
    mu_xs = Float64[]
    lambda_xs = Float64[]
    tau_xs = Float64[]
    for i in 1:n, j in 1:n, k in 1:n
        t = O2X_map[i, j, k]
        if !isnan(t[1])
            push!(mu_xs, t[1])
            push!(lambda_xs, t[2])
            push!(tau_xs, t[3])
        end
    end
    if isempty(mu_xs)
        return (NaN, NaN, NaN)
    end
    min_mu_x, max_mu_x = minimum(mu_xs), maximum(mu_xs)
    min_lambda_x, max_lambda_x = minimum(lambda_xs), maximum(lambda_xs)
    min_tau_x, max_tau_x = minimum(tau_xs), maximum(tau_xs)

    if !(min_mu_x <= mu_x <= max_mu_x &&
         min_lambda_x <= lambda_x <= max_lambda_x &&
         min_tau_x <= tau_x <= max_tau_x)
        return (NaN, NaN, NaN)
    end

    # Loop over voxels
    for i in 1:(n - 1), j in 1:(n - 1), k in 1:(n - 1)
        # Check valid
        valid = true
        for ii in 0:1, jj in 0:1, kk in 0:1
            if isnan(O2X_map[i + ii, j + jj, k + kk][1])
                valid = false
                break
            end
        end
        if !valid
            continue
        end

        # Build vals
        mu_vals = zeros(2, 2, 2)
        lambda_vals = zeros(2, 2, 2)
        tau_vals = zeros(2, 2, 2)
        for ii in 0:1, jj in 0:1, kk in 0:1
            t = O2X_map[i + ii, j + jj, k + kk]
            mu_vals[ii + 1, jj + 1, kk + 1] = t[1]
            lambda_vals[ii + 1, jj + 1, kk + 1] = t[2]
            tau_vals[ii + 1, jj + 1, kk + 1] = t[3]
        end

        # Quick AABB check
        if !(minimum(mu_vals) <= mu_x <= maximum(mu_vals) &&
             minimum(lambda_vals) <= lambda_x <= maximum(lambda_vals) &&
             minimum(tau_vals) <= tau_x <= maximum(tau_vals))
            continue
        end

        # Newton-Raphson
        params = [0.5, 0.5, 0.5]
        tol = 1e-6
        maxiter = 20
        for _ in 1:maxiter
            xd, yd, zd = params

            # Residual
            rm = trilinear_normalized(xd, yd, zd, mu_vals) - mu_x
            rl = trilinear_normalized(xd, yd, zd, lambda_vals) - lambda_x
            rt = trilinear_normalized(xd, yd, zd, tau_vals) - tau_x
            r = [rm, rl, rt]
            res_norm = norm(r)
            if res_norm < tol
                # Check bounds
                if all(0 .<= params .<= 1)
                    # Compute pre-image
                    mu_o = muo_grid[i] * (1 - params[1]) + muo_grid[i + 1] * params[1]
                    lambda_o = lambdao_grid[j] * (1 - params[2]) + lambdao_grid[j + 1] * params[2]
                    tau_o = tauo_grid[k] * (1 - params[3]) + tauo_grid[k + 1] * params[3]
                    return (mu_o, lambda_o, tau_o)
                end
                break
            end

            # Jacobian
            J = zeros(3, 3)

            # mu row
            diff_yz_mu = zeros(2, 2)
            diff_yz_mu[1, 1] = mu_vals[2, 1, 1] - mu_vals[1, 1, 1]
            diff_yz_mu[1, 2] = mu_vals[2, 1, 2] - mu_vals[1, 1, 2]
            diff_yz_mu[2, 1] = mu_vals[2, 2, 1] - mu_vals[1, 2, 1]
            diff_yz_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[1, 2, 2]
            J[1, 1] = bilinear_normalized(yd, zd, diff_yz_mu)

            diff_xz_mu = zeros(2, 2)
            diff_xz_mu[1, 1] = mu_vals[1, 2, 1] - mu_vals[1, 1, 1]
            diff_xz_mu[2, 1] = mu_vals[2, 2, 1] - mu_vals[2, 1, 1]
            diff_xz_mu[1, 2] = mu_vals[1, 2, 2] - mu_vals[1, 1, 2]
            diff_xz_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[2, 1, 2]
            J[1, 2] = bilinear_normalized(xd, zd, diff_xz_mu)

            diff_xy_mu = zeros(2, 2)
            diff_xy_mu[1, 1] = mu_vals[1, 1, 2] - mu_vals[1, 1, 1]
            diff_xy_mu[2, 1] = mu_vals[2, 1, 2] - mu_vals[2, 1, 1]
            diff_xy_mu[1, 2] = mu_vals[1, 2, 2] - mu_vals[1, 2, 1]
            diff_xy_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[2, 2, 1]
            J[1, 3] = bilinear_normalized(xd, yd, diff_xy_mu)

            # lambda row
            diff_yz_lambda = zeros(2, 2)
            diff_yz_lambda[1, 1] = lambda_vals[2, 1, 1] - lambda_vals[1, 1, 1]
            diff_yz_lambda[1, 2] = lambda_vals[2, 1, 2] - lambda_vals[1, 1, 2]
            diff_yz_lambda[2, 1] = lambda_vals[2, 2, 1] - lambda_vals[1, 2, 1]
            diff_yz_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[1, 2, 2]
            J[2, 1] = bilinear_normalized(yd, zd, diff_yz_lambda)

            diff_xz_lambda = zeros(2, 2)
            diff_xz_lambda[1, 1] = lambda_vals[1, 2, 1] - lambda_vals[1, 1, 1]
            diff_xz_lambda[2, 1] = lambda_vals[2, 2, 1] - lambda_vals[2, 1, 1]
            diff_xz_lambda[1, 2] = lambda_vals[1, 2, 2] - lambda_vals[1, 1, 2]
            diff_xz_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[2, 1, 2]
            J[2, 2] = bilinear_normalized(xd, zd, diff_xz_lambda)

            diff_xy_lambda = zeros(2, 2)
            diff_xy_lambda[1, 1] = lambda_vals[1, 1, 2] - lambda_vals[1, 1, 1]
            diff_xy_lambda[2, 1] = lambda_vals[2, 1, 2] - lambda_vals[2, 1, 1]
            diff_xy_lambda[1, 2] = lambda_vals[1, 2, 2] - lambda_vals[1, 2, 1]
            diff_xy_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[2, 2, 1]
            J[2, 3] = bilinear_normalized(xd, yd, diff_xy_lambda)

            # tau row
            diff_yz_tau = zeros(2, 2)
            diff_yz_tau[1, 1] = tau_vals[2, 1, 1] - tau_vals[1, 1, 1]
            diff_yz_tau[1, 2] = tau_vals[2, 1, 2] - tau_vals[1, 1, 2]
            diff_yz_tau[2, 1] = tau_vals[2, 2, 1] - tau_vals[1, 2, 1]
            diff_yz_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[1, 2, 2]
            J[3, 1] = bilinear_normalized(yd, zd, diff_yz_tau)

            diff_xz_tau = zeros(2, 2)
            diff_xz_tau[1, 1] = tau_vals[1, 2, 1] - tau_vals[1, 1, 1]
            diff_xz_tau[2, 1] = tau_vals[2, 2, 1] - tau_vals[2, 1, 1]
            diff_xz_tau[1, 2] = tau_vals[1, 2, 2] - tau_vals[1, 1, 2]
            diff_xz_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[2, 1, 2]
            J[3, 2] = bilinear_normalized(xd, zd, diff_xz_tau)

            diff_xy_tau = zeros(2, 2)
            diff_xy_tau[1, 1] = tau_vals[1, 1, 2] - tau_vals[1, 1, 1]
            diff_xy_tau[2, 1] = tau_vals[2, 1, 2] - tau_vals[2, 1, 1]
            diff_xy_tau[1, 2] = tau_vals[1, 2, 2] - tau_vals[1, 2, 1]
            diff_xy_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[2, 2, 1]
            J[3, 3] = bilinear_normalized(xd, yd, diff_xy_tau)

            # Update
            delta = J \ (-r)
            params .+= delta
            params = clamp.(params, 0.0, 1.0)
        end

        # Final check
        xd, yd, zd = params
        rm = trilinear_normalized(xd, yd, zd, mu_vals) - mu_x
        rl = trilinear_normalized(xd, yd, zd, lambda_vals) - lambda_x
        rt = trilinear_normalized(xd, yd, zd, tau_vals) - tau_x
        final_res = norm([rm, rl, rt])
        if final_res < tol && all(0 .<= params .<= 1)
            mu_o = muo_grid[i] * (1 - xd) + muo_grid[i + 1] * xd
            lambda_o = lambdao_grid[j] * (1 - yd) + lambdao_grid[j + 1] * yd
            tau_o = tauo_grid[k] * (1 - zd) + tauo_grid[k + 1] * zd
            return (mu_o, lambda_o, tau_o)
        end
    end

    return (NaN, NaN, NaN)
end