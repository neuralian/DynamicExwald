function threshold_crossing_figure(;
    tau::Float64    = 0.05,
    q::Float64      = 0.5,
    sigma::Float64  = 0.8,
    a::Float64      = 0.6,
    dt::Float64     = DEFAULT_SIMULATION_DT,
    T::Float64      = 0.5,
    seed::Int       = 42,
    n_traces::Int   = 5)

    Random.seed!(seed)

    # ── Time vector ──────────────────────────────────────────────────────────
    t = collect(dt:dt:T)
    n = length(t)

    # ── Run n_traces realisations ────────────────────────────────────────────
    # Store voltage traces up to first spike for each realisation
    traces      = Vector{Vector{Float64}}()
    spike_times = Float64[]

    for rep in 1:n_traces
        neuron, _ = make_qSLIFc_neuron(a, sigma, tau, q; dt=dt)
        V_trace   = Float64[]
        for k in 1:n
            fired, V = neuron(_ -> 0.0, t[k])
            push!(V_trace, V)
            if fired
                push!(spike_times, t[k])
                break
            end
        end
        push!(traces, V_trace)
    end

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = Figure(size=(900, 500))
    ax  = Axis(fig[1,1],
               xlabel = "Time (s)",
               ylabel = "Membrane potential (normalised)",
               title  = @sprintf("Fractional LIF  (q=%.2f, τ=%.0f ms, σ=%.2f)",
                                  q, tau*1000, sigma))

    # Threshold line
    hlines!(ax, [1.0]; color=:black, linewidth=1.5, linestyle=:dash, label="Threshold")

    # Reset level
    hlines!(ax, [0.0]; color=(:gray, 0.4), linewidth=1.0, linestyle=:dot)

    # Voltage traces — colour by which crosses threshold first
    colors = Makie.wong_colors()
    for (i, V) in enumerate(traces)
        ti = t[1:length(V)]
        lines!(ax, ti, V;
               color     = (colors[mod1(i, length(colors))], 0.7),
               linewidth = 1.5)

        # Mark spike with vertical line and dot
        if i <= length(spike_times)
            st = spike_times[i]
            vlines!(ax, [st];
                    color     = (colors[mod1(i, length(colors))], 0.5),
                    linewidth = 1.0,
                    linestyle = :dot)
            scatter!(ax, [st], [1.0];
                     color      = colors[mod1(i, length(colors))],
                     markersize = 10)
        end
    end

    # Noise band illustration — shaded region around zero
    noise_band = sigma * sqrt(dt) * 3
    band!(ax, t, fill(-noise_band, n), fill(noise_band, n);
          color = (:gray, 0.15))

    axislegend(ax; position=:lt)
    display(fig)
    return fig
end