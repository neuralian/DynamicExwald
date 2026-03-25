using GLMakie
using Statistics
using Printf

"""
    lif_spontaneous_spike_train(mu, lambda, tau; dt=1e-4, T=100.0)

LIF neuron parameterised via Inverse Gaussian (mu, lambda) and time constant tau.

In the drift-diffusion limit (tau → ∞) the ISIs follow IG(mu, lambda) exactly.

# Arguments
- `mu`     : Target mean ISI [s]
- `lambda` : IG shape parameter (larger = lower variance)
- `tau`    : Membrane time constant [s] = C/g

# Keywords
- `dt`  : Time step [s] (default 1e-4)
- `T`   : Total simulation time [s] (default 100.0)

# Returns
- `intervals` : Vector of ISIs [s]
- `(C, g, s)` : NamedTuple of internal biophysical parameters
"""
function lif_spontaneous_spike_train(
    mu::Float64,
    lambda::Float64,
    tau::Float64;
    dt::Float64 = 1e-4,
    T::Float64  = 1000.0
)
    V_reset = 0.0
    V_th    = 1.0
    θ       = V_th - V_reset    # = 1.0

    # ── Drift-diffusion parameters ───────────────────────────────────────────
    # For pure drift-diffusion  dV = a·dt + σ_v·dW,  FPT from 0 → θ:
    #
    #   E[T]   = θ/a                 = mu      →  a   = θ/mu  = 1/mu
    #   Var[T] = σ_v²·θ / a³        = mu³/λ
    #
    # Solving for σ_v:
    #   σ_v² = Var[T]·a³/θ = (mu³/λ)·(1/mu³)/1 = 1/λ
    #   σ_v  = 1/√λ                              ← key fix: independent of mu!
    #
    a   = θ / mu          # drift [norm-V/s]
    σ_v = 1.0 / sqrt(lambda)   # noise std [norm-V/√s]

    # ── Biophysical parameters ───────────────────────────────────────────────
    # LIF in normalised voltage, leak reversal at V_reset:
    #   dV = [a - V/tau] dt + σ_v·dW
    # where the bias sets drift at reset = a, and leak pulls back for V > 0.
    #
    # Physical units: C = 1 nF (arbitrary scale), g = C/tau, s = C·σ_v
    C = 1e-9
    g = C / tau
    s = C * σ_v

    # ── Euler-Maruyama integration ───────────────────────────────────────────
    n_steps   = round(Int, T / dt)
    sqrt_dt   = sqrt(dt)
    V         = V_reset
    t_last    = 0.0
    intervals = Float64[]

    for k in 1:n_steps
        dV = (a - V / tau) * dt + σ_v * sqrt_dt * randn()
        V += dV
        if V >= V_th
            push!(intervals, k * dt - t_last)
            t_last = k * dt
            V = V - V_th  #V_reset
        end
    end

    if isempty(intervals)
        @warn "No spikes generated — try larger T or smaller mu."
        return intervals, (C=C, g=g, s=s)
    end

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = Figure(size = (820, 520))
    ax  = Axis(fig[1,1],
        xlabel = "Interspike interval (s)",
        ylabel = "Probability density",
        title  = "LIF ISI  (μ=$(mu) s,  λ=$(lambda),  " *
                 "τ=$(round(tau*1e3, digits=2)) ms,  n=$(length(intervals)))"
    )

    hist!(ax, intervals;
          normalization = :pdf, bins = 32,
          color = (:steelblue, 0.65), strokewidth = 0.4)

    x_hi = 2*min(4*mu + 4*sqrt(mu^3/lambda), maximum(intervals))
    xs   = range(1e-7, x_hi, length = 1000)
    ig_pdf(x) = sqrt(lambda / (2π * x^3)) *
                exp(-lambda * (x - mu)^2 / (2 * mu^2 * x))
    lines!(ax, collect(xs), ig_pdf.(xs);
           color = :crimson, linewidth = 2.5, label = "IG(μ,λ) theory")
    axislegend(ax; position = :rt)
    display(fig)

    # ── Summary ──────────────────────────────────────────────────────────────
    println("Spikes : $(length(intervals))")
    @printf("  Mean ISI : %.6f s   (target μ = %.6f)\n", mean(intervals), mu)
    @printf("  Var  ISI : %.8f   (IG theory: %.8f)\n", var(intervals), mu^3/lambda)
    println("Internal parameters:")
    @printf("  C = %.3f nF\n", C*1e9)
    @printf("  g = %.6f nS   (tau = %.3f ms)\n", g*1e9, tau*1e3)
    @printf("  s = %.6f pA/√s\n", s*1e12)

    return intervals, (C=C, g=g, s=s)
end