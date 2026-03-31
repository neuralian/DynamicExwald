"""
    fit_qSLIF_to_Exwald_v2(mu_target, lambda_target, tau_target; ...)

Fit qSLIF to target Exwald using reparameterised excitability ratio ρ = a·tau
to avoid degenerate solutions where tau is too small.

Optimises over (mu_s, lambda_s, φ, q) where:
  ρ     = 1 + exp(φ)          — excitability ratio, guaranteed > 1
  tau_s = mu_s / (-log(1-1/ρ))  — derived from ρ and mu_s
  a     = ρ / tau_s            — derived from ρ and tau_s
"""
function fit_qSLIF_to_Exwald_v2(
    mu_target::Float64,
    lambda_target::Float64,
    tau_target::Float64;
    N::Int        = 2000,
    dt::Float64   = DEFAULT_SIMULATION_DT,
    n_starts::Int = 5,
    seed::Int     = 42,
    verbose::Bool = true
)
    # ── Target distribution statistics ──────────────────────────────────────
    mean_target = mu_target + tau_target
    var_target  = mu_target^3 / lambda_target + tau_target^2
    cv_target   = sqrt(var_target) / mean_target

    # Precompute timeout from target distribution statistics
    timeout = mean_target + 10.0 * sqrt(var_target)

    # ── Reparameterisation ───────────────────────────────────────────────────
    # Optimise over (mu_s, lambda_s, phi, q) where:
    #   rho     = 1 + exp(phi)                 excitability ratio > 1
    #   tau_s   = mu_s / (-log(1 - 1/rho))    membrane time constant
    #   a       = rho / tau_s                  SDE drift coefficient
    #   sigma   = 1 / sqrt(lambda_s)           SDE noise coefficient
    function decode(mu_s, lambda_s, phi, q)
        rho   = 1.0 + exp(phi)
        tau_s = mu_s / (-log(1.0 - 1.0 / rho))
        a     = rho / tau_s
        sigma = 1.0 / sqrt(lambda_s)
        return a, sigma, tau_s, q, rho
    end

    # ── Bounds in (mu_s, lambda_s, phi, q) space ────────────────────────────
    phi_lo = log(0.01)    # rho = 1.01
    phi_hi = log(99.0)    # rho = 100
    LB = [1e-4,  0.1,  phi_lo,  0.1]
    UB = [10.0,  100.0, phi_hi,  1.0]

    # ── Initial guess generator ──────────────────────────────────────────────
    function make_initial_guess(rng)
        ε         = clamp(cv_target^2, 0.05, 0.95)
        mu_s0     = mean_target * (1.0 - ε)
        lambda_s0 = mu_s0 / cv_target^2
        rho0      = 2.0 + 3.0 * rand(rng)
        phi0      = log(rho0 - 1.0)
        q0        = 0.5 + 0.5 * rand(rng)
        return [mu_s0, lambda_s0, phi0, q0]
    end

    # ── Loss function ────────────────────────────────────────────────────────
    call_count = Ref(0)

    function loss(param, _)
        mu_s, lambda_s, phi, q = param
        (mu_s <= 0.0 || lambda_s <= 0.0 || q <= 0.0 || q > 1.0) && return 1e6

        a, sigma, tau_s, q, rho = decode(mu_s, lambda_s, phi, q)

        tau_s > 100.0 * mean_target && return 1e6

        call_count[] += 1
        Random.seed!(seed + call_count[])

        # Run qSLIF simulation
        T      = N * mean_target * 3.0
        neuron, _ = make_qSLIFc_neuron(a, sigma, tau_s, q; dt=dt)

        intervals = Float64[]
        t         = 0.0
        t_last    = 0.0
        n_spikes  = 0

        while n_spikes < N && t < T
            t += dt
            if neuron(_ -> 0.0, t)
                Delta_t = t - t_last
                if Delta_t > timeout
                    break
                end
                push!(intervals, Delta_t)
                t_last   = t
                n_spikes += 1
            end
        end

        length(intervals) < N ÷ 2 && return 1e6

        # qSLIFparam vector matches make_qSLIFc_neuron signature
        qSLIFparam = [a, sigma, tau_s, q]
        Exwaldparam = [mu_target, lambda_target, tau_target]
        kld = qSLIF2Exwald_KLD(qSLIFparam, Exwaldparam, N, dt)
        l = kld

        if verbose
            @printf("  ρ=%6.3f  τ=%8.5f  a=%8.4f  σ=%7.4f  q=%.3f  loss=%.5f\n",
                    rho, tau_s, a, sigma, q, l)
        end
        return l
    end

    # ── Multi-start optimisation ─────────────────────────────────────────────
    best_loss   = Inf
    best_params = nothing
    rng         = Random.MersenneTwister(seed)

    for i in 1:n_starts
        pInit = clamp.(make_initial_guess(rng), LB .+ 1e-6, UB .- 1e-6)
        verbose && println("\nStart $i/$n_starts  pInit=$pInit")

        try
            f    = OptimizationFunction(loss)
            prob = OptimizationProblem(f, pInit, nothing; lb=LB, ub=UB)
            sol  = solve(prob, NLopt.LN_SBPLX();
                         reltol=1e-3, abstol=1e-3, maxiters=500)

            l = loss(sol.u, nothing)
            verbose && @printf("  → Start %d final loss = %.6f\n", i, l)

            if l < best_loss
                best_loss   = l
                best_params = sol.u
            end
        catch e
            verbose && println("  Start $i failed: $e")
        end
    end

    isnothing(best_params) && error("All optimisation starts failed")

    mu_s, lambda_s, phi, q = best_params
    a, sigma, tau_s, q, rho = decode(mu_s, lambda_s, phi, q)

    if verbose
        println("\n── Best fit ──────────────────────────────────────")
        @printf("  a        = %.5f\n",  a)
        @printf("  sigma    = %.5f\n",  sigma)
        @printf("  tau_s    = %.5f s\n", tau_s)
        @printf("  q        = %.4f\n",   q)
        @printf("  rho      = %.3f\n",   rho)
        @printf("  loss     = %.6f\n",   best_loss)
    end

    return (a=a, sigma=sigma, tau_s=tau_s, q=q, rho=rho, loss=best_loss)
end