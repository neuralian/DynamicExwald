
using LinearAlgebra  # Not strictly needed, but for potential future use

"""
    fpt_ou_pdf(t, x0, kappa, mu, sigma, b; N=2000)

Compute the probability density function of the first passage time for an
Ornstein-Uhlenbeck process dX_t = kappa * (mu - X_t) * dt + sigma * dW_t,
starting at X_0 = x0, to hit the barrier b.

This uses the semi-analytical method of Lipton & Kaushansky (2018) involving
solution of a Volterra integral equation via discretization. Handles both upper
and lower barriers via reflection symmetry.

# Arguments
- `t::Real`: Time at which to evaluate the density (>0).
- `x0::Real`: Starting value.
- `kappa::Real >0`: Reversion speed.
- `mu::Real`: Long-term mean.
- `sigma::Real >0`: Volatility.
- `b::Real`: Barrier level.
- `N::Int=2000`: Number of grid points for discretization (higher for accuracy).

# Returns
- `Float64`: The FPT density g(t) = d/dt P(τ_b ≤ t).

Note: For barrier at mu (special case), a closed-form exists but is not used here.
"""
function fpt_ou_pdf(t::Real, x0::Real, kappa::Real, mu::Real, sigma::Real, b::Real; N::Int=2000)
    if t <= 0
        return 0.0
    end
    if kappa <= 0 || sigma <= 0
        error("kappa and sigma must be positive")
    end

    alpha = sqrt(kappa) / sigma
    y0 = alpha * (x0 - mu)
    c = alpha * (b - mu)

    t_bar = kappa * t
    if y0 > c
        g_bar = ou_fpt_density_std(t_bar, y0, c, N=N)
    else
        g_bar = ou_fpt_density_std(t_bar, -y0, -c, N=N)
    end
    return kappa * g_bar
end

"""
    ou_fpt_density_std(t, z, b; N=2000)

Standardized version: density for dX_t = -X_t dt + dW_t, first hitting time to b
from z > b (handles b of either sign via the general Volterra formulation).
"""
function ou_fpt_density_std(t::Real, z::Real, b::Real; N::Int=2000)
    if t <= 0
        return 0.0
    end
    if b >= z
        error("In standardized case, z > b required")
    end

    et = exp(t)
    e2t = et * et
    denom = e2t - 1.0
    tau = denom / 2.0

    # Term 1: Free (image) term
    diffbz = et * b - z
    exp_arg = - (diffbz^2) / denom + 2.0 * t
    term1 = - diffbz * exp(exp_arg) / sqrt(pi * denom^3)

    # Discretize [0, tau] with N intervals, N+1 points
    h = tau / N
    taus = [k * h for k in 0:N]
    nu = zeros(N + 1)
    nu[1] = 0.0  # At tau=0

    # Solve Volterra: nu(tau) = -free(tau) - ∫_0^tau K(tau, s) nu(s) ds
    # Using left rectangle rule for integral
    for i in 2:(N + 1)  # i=1 is 0
        ti = taus[i]
        bt_i = b * sqrt(2 * ti + 1.0)
        free_i = exp( - (bt_i - z)^2 / (2 * ti) ) / sqrt(2 * pi * ti )

        integ = 0.0
        for j in 1:(i - 1)
            tj = taus[j]
            bt_j = b * sqrt(2 * tj + 1.0)
            delta_bt = bt_i - bt_j
            delta_t = ti - tj
            exp_k = exp( - delta_bt^2 / (2 * delta_t) )
            k_j = delta_bt * exp_k / sqrt(2 * pi * delta_t^3)
            integ += h * k_j * nu[j]
        end
        nu[i] = - free_i - integ
    end

    nu_tau = nu[end]

    # Term 2: Local term with nu(tau)
    term2 = - (et * b + e2t / sqrt(pi * denom)) * nu_tau

    # Term 3: Integral term
    term3 = 0.0
    coef3 = e2t / sqrt(8 * pi)
    for j in 1:N  # j=1 to N, taus[N+1]=tau, but nu[N+1]=nu_tau
        tj = taus[j]
        delta_t = tau - tj
        if delta_t <= 0
            continue
        end
        nu_diff = nu[j] - nu_tau
        bracket = 1.0 - 4.0 * b^2 * delta_t
        exp_part = exp(-2.0 * b^2 * delta_t)
        integrand_j = bracket * exp_part * nu_diff / (delta_t ^ 1.5)
        term3 += h * integrand_j
    end
    term3 *= coef3

    return term1 + term2 + term3
end

"""
    fpt_ig_leaky_pdf(t, mu_ig, lambda, tau; N=2000)

Compute the FPT density for a leaky integrate-and-fire model (OU process), parameterized
by the inverse Gaussian parameters (mu_ig, lambda) for the non-leaky case (tau = ∞)
and the reversion time constant tau > 0.

- Starts at x0 = 0, hits upper barrier b = 1.
- When tau → ∞ (kappa → 0), recovers IG(mu_ig, lambda) density.
- kappa = 1 / tau (reversion speed).
- sigma = 1 / sqrt(lambda) (volatility).
- mu_ou = tau / mu_ig (long-term mean, scaled to match IG drift nu = 1 / mu_ig).

# Arguments
- `t::Real`: Time (>0).
- `mu_ig::Real >0`: Mean of the limiting IG distribution.
- `lambda::Real >0`: Shape of the limiting IG distribution.
- `tau::Real >0`: Reversion time constant.
- `N::Int=2000`: Discretization points.

# Returns
- `Float64`: FPT density at t.
"""
function fpt_ig_leaky_pdf(t::Real, mu_ig::Real, lambda::Real, tau::Real; N::Int=2000)
    if t <= 0
        return 0.0
    end
    if mu_ig <= 0 || lambda <= 0 || tau <= 0
        error("mu_ig, lambda, tau must be positive")
    end

    kappa = 1.0 / tau
    sigma = 1.0 / sqrt(lambda)
    x0 = 0.0
    b = 1.0
    mu_ou = tau / mu_ig

    return fpt_ou_pdf(t, x0, kappa, mu_ou, sigma, b; N=N)
end

# Optional: Limiting IG density for verification (tau = ∞)
"""
    ig_pdf(t, mu_ig, lambda)

Inverse Gaussian density: limiting case as tau → ∞.
"""
function ig_pdf(t::Real, mu_ig::Real, lambda::Real)
    if t <= 0
        return 0.0
    end
    return sqrt(lambda / (2 * pi * t^3)) * exp( -lambda * (t - mu_ig)^2 / (2 * mu_ig^2 * t) )
end