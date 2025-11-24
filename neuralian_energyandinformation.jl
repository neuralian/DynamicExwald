"""
Energy cost components of a neuron
All values in ATP molecules per second
"""
struct NeuronEnergyBudget
    baseline::Float64        # Resting/housekeeping
    per_spike::Float64       # Cost per action potential
    synaptic_base::Float64   # Baseline synaptic activity
    per_synapse::Float64     # Cost per synaptic event
end

# Typical cortical pyramidal neuron
function cortical_neuron_energy()
    return NeuronEnergyBudget(
        4.7e9,      # ~4.7 billion ATP/s baseline (resting)
        1e7,        # ~10 million ATP per spike
        2.3e9,      # ~2.3 billion ATP/s synaptic baseline
        1.6e5       # ~160,000 ATP per synaptic event
    )
end

using CairoMakie

"""
Compute total energy consumption as function of firing rate
"""
function energy_consumption(firing_rate_hz, budget::NeuronEnergyBudget;
                           synapse_rate_multiplier=100)
    # Baseline energy (independent of firing)
    E_baseline = budget.baseline
    
    # Spiking energy (scales with firing rate)
    E_spiking = budget.per_spike * firing_rate_hz
    
    # Synaptic energy (baseline + activity-dependent)
    # Assume synapse rate scales with firing rate
    synapse_rate = synapse_rate_multiplier * firing_rate_hz
    E_synaptic = budget.synaptic_base + budget.per_synapse * synapse_rate
    
    return (
        total = E_baseline + E_spiking + E_synaptic,
        baseline = E_baseline,
        spiking = E_spiking,
        synaptic = E_synaptic
    )
end

function plot_neuron_energybudget()

    # Plot energy vs firing rate
    budget = cortical_neuron_energy()
    rates = 0:0.5:100  # 0 to 100 Hz

    energies = [energy_consumption(r, budget) for r in rates]
    E_total = [e.total for e in energies]
    E_baseline = [e.baseline for e in energies]
    E_spiking = [e.spiking for e in energies]
    E_synaptic = [e.synaptic for e in energies]

    fig = Figure(size=(1400, 900))

    # Total energy
    ax1 = Axis(fig[1, 1], 
            xlabel="Firing Rate (Hz)", 
            ylabel="Energy (ATP/s)",
            title="Total Energy Consumption")

    lines!(ax1, rates, E_total ./ 1e9, linewidth=3, 
        color=:black, label="Total")
    lines!(ax1, rates, E_baseline ./ 1e9, linewidth=2,
        linestyle=:dash, color=:blue, label="Baseline")
    lines!(ax1, rates, E_spiking ./ 1e9, linewidth=2,
        color=:red, label="Spiking")
    lines!(ax1, rates, E_synaptic ./ 1e9, linewidth=2,
        color=:green, label="Synaptic")

    ax1.ylabel = "Energy (billion ATP/s)"
    axislegend(ax1, position=:lt)

    # Breakdown by component
    ax2 = Axis(fig[1, 2],
            xlabel="Firing Rate (Hz)",
            ylabel="Fraction of Total Energy",
            title="Energy Budget Breakdown")

    frac_baseline = E_baseline ./ E_total
    frac_spiking = E_spiking ./ E_total
    frac_synaptic = E_synaptic ./ E_total

    band!(ax2, rates, zeros(length(rates)), frac_baseline,
        color=(:blue, 0.5), label="Baseline")
    band!(ax2, rates, frac_baseline, frac_baseline .+ frac_spiking,
        color=(:red, 0.5), label="Spiking")
    band!(ax2, rates, frac_baseline .+ frac_spiking, ones(length(rates)),
        color=(:green, 0.5), label="Synaptic")

    axislegend(ax2, position=:rt)

    # Energy per spike
    ax3 = Axis(fig[2, 1],
            xlabel="Firing Rate (Hz)",
            ylabel="Energy per Spike (million ATP)",
            title="Average Energy Cost per Spike")

    E_per_spike = E_total ./ (rates .+ 0.001)  # Avoid division by zero
    lines!(ax3, rates[2:end], E_per_spike[2:end] ./ 1e6, linewidth=3)
    hlines!(ax3, [budget.per_spike / 1e6], 
            linestyle=:dash, color=:red, label="Direct spike cost")

    axislegend(ax3)

    # Energy efficiency
    ax4 = Axis(fig[2, 2],
            xlabel="Firing Rate (Hz)",
            ylabel="Information / Energy",
            title="Energy Efficiency (conceptual)")

    # Assuming information scales with firing rate (simplified)
    # But energy increases faster
    info_rate = rates  # bits/s (very simplified!)
    efficiency = info_rate ./ (E_total ./ 1e9)

    lines!(ax4, rates[2:end], efficiency[2:end], linewidth=3)

    fig

end

"""
Energy cost of single action potential
"""
function spike_energy_cost(; 
    na_ions_in=10e9,           # Na+ ions entering per spike
    k_ions_out=10e9,           # K+ ions leaving per spike
    na_k_pump_ratio=3          # 3 Na+ out : 2 K+ in per ATP
)
    # Each ATP pumps 3 Na+ out and 2 K+ in
    # Need to pump out Na+ that entered
    atp_for_na = na_ions_in / na_k_pump_ratio
    
    # And pump in K+ that left
    atp_for_k = k_ions_out / 2  # 2 K+ per ATP
    
    # Total (simplified - they're coupled)
    total_atp = max(atp_for_na, atp_for_k)
    
    return total_atp
end

function show_spike_energy_cost()

    cost = spike_energy_cost()
    println("Energy per spike: $(cost/1e6) million ATP")
    println("At 10 Hz: $(cost * 10 / 1e9) billion ATP/s")
    println("At 100 Hz: $(cost * 100 / 1e9) billion ATP/s")

end

"""
Synaptic energy cost
"""
function synaptic_energy(firing_rate_hz, 
                        n_synapses=10000,
                        vesicles_per_spike=1)
    # Energy per synaptic vesicle cycle
    E_per_vesicle = 1.6e5  # ATP
    
    # Number of vesicle releases
    releases_per_sec = firing_rate_hz * n_synapses * vesicles_per_spike
    
    # Total synaptic energy
    E_synaptic = E_per_vesicle * releases_per_sec
    
    return E_synaptic
end

function show_synaptic_energy_cost()
println("Synaptic energy at different rates:")
    for rate in [1, 10, 50, 100]
        E = synaptic_energy(rate)
        println("  $(rate) Hz: $(round(E/1e9, digits=2)) billion ATP/s")
    end
end

function show_power_consumption()

"""
Empirical relationship: E ≈ E₀ + α·f
where f is firing rate
"""
function energy_linear_fit(firing_rate_hz)
    E₀ = 7e9        # Baseline (billion ATP/s)
    α = 5e7         # Slope (ATP per spike)
    
    return E₀ + α * firing_rate_hz
end

# Plot
rates = 0:100
E_linear = energy_linear_fit.(rates)

fig, ax = lines(rates, E_linear ./ 1e9,
               axis=(xlabel="Firing Rate (Hz)",
                    ylabel="Energy (billion ATP/s)",
                    title="Linear Approximation: E = E₀ + α·f"))

end

"""
Non-linear energy cost at high rates
"""
function energy_nonlinear(firing_rate_hz)
    E₀ = 7e9
    α = 5e7
    β = 1e5  # Non-linear term
    
    return E₀ + α * firing_rate_hz + β * firing_rate_hz^2
end

struct NeuronType
    name::String
    baseline_energy::Float64      # ATP/s
    energy_per_spike::Float64     # ATP/spike
    typical_rate::Float64         # Hz
end

function compare_power_by_neurontype()

    neuron_types = [
        NeuronType("Cortical pyramidal", 4.7e9, 1e7, 5),
        NeuronType("Cortical interneuron", 2e9, 8e6, 20),
        NeuronType("Cerebellar Purkinje", 6e9, 1.2e7, 50),
        NeuronType("Vestibular afferent", 3e9, 1e7, 80),
        NeuronType("Thalamic relay", 3.5e9, 9e6, 10),
    ]

    println("Energy consumption by neuron type:")
    println("="^70)
    for nt in neuron_types
        E_rest = nt.baseline_energy
        E_active = E_rest + nt.energy_per_spike * nt.typical_rate
        
        println(rpad(nt.name, 25), 
                "Rest: $(rpad(round(E_rest/1e9, digits=1), 5)) billion ATP/s, ",
                "Active: $(round(E_active/1e9, digits=1)) billion ATP/s @ $(nt.typical_rate) Hz")
    end

end

"""
Estimate energy efficiency: information per ATP
Very simplified - assumes spikes carry ~1 bit
"""
function energy_efficiency(firing_rate_hz, energy_budget)
    E = energy_consumption(firing_rate_hz, energy_budget).total
    
    # Information rate (simplified)
    # Assume each spike carries ~1 bit
    info_rate = firing_rate_hz  # bits/s
    
    # Efficiency: bits per ATP
    efficiency = info_rate / E
    
    return efficiency
end

function show_bits_per_ATP()

budget = cortical_neuron_energy()
rates = 1:100

efficiencies = [energy_efficiency(r, budget) for r in rates]

fig, ax = lines(rates, efficiencies .* 1e9,
               axis=(xlabel="Firing Rate (Hz)",
                    ylabel="Information (bits) per billion ATP",
                    title="Energy Efficiency vs Firing Rate"))

# Find optimal
optimal_idx = argmax(efficiencies)
optimal_rate = rates[optimal_idx]
vlines!(ax, [optimal_rate], linestyle=:dash, 
        label="Optimal ≈ $(optimal_rate) Hz")
axislegend(ax)

fig

end

function summarize_energy_cost_of_spiking()

    budget = cortical_neuron_energy()

    # At 0 Hz
    E_rest = energy_consumption(0, budget).total

    # At 10 Hz (typical cortical)
    E_10hz = energy_consumption(10, budget).total

    println("Energy consumption:")
    println("  Resting: $(round(E_rest/1e9, digits=1)) billion ATP/s")
    println("  10 Hz: $(round(E_10hz/1e9, digits=1)) billion ATP/s")
    println("  Increase: $(round(100*(E_10hz/E_rest - 1), digits=1))%")


    # At high firing rates, synaptic costs dominate

    for rate in [0, 10, 50, 100]
        E = energy_consumption(rate, budget)
        
        println("\nAt $(rate) Hz:")
        println("  Baseline: $(round(100*E.baseline/E.total, digits=1))%")
        println("  Spiking: $(round(100*E.spiking/E.total, digits=1))%")
        println("  Synaptic: $(round(100*E.synaptic/E.total, digits=1))%")
    end

end

# Empirical scaling from Attwell & Laughlin (2001)
# For mammalian cortex:

# Total brain energy ~ 20% of body energy
# Signaling (spikes + synapses) ~ 80% of brain energy
# At rest: ~50% goes to maintaining resting potential
# During activity: ~50% goes to synaptic transmission

"""
Compare dense vs sparse coding
"""
function compare_coding_efficiency(n_neurons::Int64=10000)
    budget = cortical_neuron_energy()
    
    # Dense coding: all neurons fire at moderate rate
    rate_dense = 10.0  # Hz
    E_dense = n_neurons * energy_consumption(rate_dense, budget).total
    
    # Sparse coding: few neurons fire at high rate
    rate_sparse = 50.0  # Hz
    fraction_active = 0.1
    n_active = Int(floor(n_neurons * fraction_active))
    E_sparse = (n_active * energy_consumption(rate_sparse, budget).total +
                (n_neurons - n_active) * energy_consumption(0, budget).total)
    
    println("Coding efficiency comparison ($n_neurons neurons):")
    println("  Dense (all @ $(rate_dense) Hz): $(round(E_dense/1e12, digits=1)) trillion ATP/s")
    println("  Sparse ($(fraction_active*100)% @ $(rate_sparse) Hz): $(round(E_sparse/1e12, digits=1)) trillion ATP/s")
    println("  Savings: $(round(100*(1 - E_sparse/E_dense), digits=1))%")
end


function compare_efficiency()
    # Vestibular afferent at 100 Hz vs cortical neuron at 5 Hz
    E_vest_100 = energy_consumption(100, cortical_neuron_energy()).total
    E_ctx_5 = energy_consumption(5, cortical_neuron_energy()).total

    println("Energy comparison:")
    println("  Vestibular @ 100 Hz: $(round(E_vest_100/1e9, digits=1)) billion ATP/s")
    println("  Cortical @ 5 Hz: $(round(E_ctx_5/1e9, digits=1)) billion ATP/s")
    println("  Ratio: $(round(E_vest_100/E_ctx_5, digits=1))x more expensive")
 
    ## Summary

    # **Key relationship:**
    # ```
    # E_total(f) ≈ E_baseline + E_spike·f + E_synapse·f

    # Where:
    # - E_baseline ≈ 5-7 billion ATP/s (constant)
    # - E_spike ≈ 10-20 million ATP per spike
    # - E_synapse ≈ 100-200 million ATP per spike (synaptic transmission)

end

"""
Energy consumption estimates with uncertainty ranges
Based on published literature (requires verification)
"""
function energy_consumption_with_uncertainty(firing_rate_hz)
    # These are APPROXIMATE - verify with primary sources!
    
    # Baseline energy (range from literature)
    E_baseline_low = 3e9   # billion ATP/s
    E_baseline_high = 8e9
    E_baseline = 5.5e9  # midpoint
    
    # Energy per spike (range from literature)
    E_spike_low = 5e6   # million ATP
    E_spike_high = 2e7
    E_spike = 1e7  # midpoint
    
    # Synaptic cost per spike (highly variable!)
    E_synapse_low = 5e7
    E_synapse_high = 3e8
    E_synapse = 1.5e8  # midpoint
    
    # Calculate
    E_total = E_baseline + (E_spike + E_synapse) * firing_rate_hz
    E_total_low = E_baseline_low + (E_spike_low + E_synapse_low) * firing_rate_hz
    E_total_high = E_baseline_high + (E_spike_high + E_synapse_high) * firing_rate_hz
    
    return (
        estimate = E_total,
        lower_bound = E_total_low,
        upper_bound = E_total_high,
        note = "VERIFY THESE VALUES WITH PRIMARY LITERATURE"
    )
end

function show_energy_consumption_with_uncertainty()
    # Example
    result = energy_consumption_with_uncertainty(10.0)
    println("At 10 Hz:")
    println("  Estimate: $(result.estimate/1e9) billion ATP/s")
    println("  Range: $(result.lower_bound/1e9) - $(result.upper_bound/1e9)")
    println("  Note: $(result.note)")

end

"""
Voltage-dependent Na+ entry during incomplete repolarization
"""
function sodium_entry_per_spike(V_start, V_threshold=-45.0, V_rest=-65.0)
    # Driving force for Na+ depends on starting voltage
    # Higher starting voltage → larger driving force → more Na+ enters
    
    # Voltage excursion
    ΔV_full = V_threshold - V_rest  # Full spike from rest
    ΔV_partial = V_threshold - V_start  # Spike from elevated baseline
    
    # Na+ entry scales with voltage excursion (approximately)
    # More accurately, scales with activation curve integral
    Na_entry_full = 1.0  # Normalized
    Na_entry_partial = ΔV_partial / ΔV_full
    
    # But activation kinetics mean more Na+ channels open when starting higher
    # Empirically, this can mean 1.3-1.5x more Na+ entry
    correction_factor = 1.0 + 0.5 * (V_start - V_rest) / (V_threshold - V_rest)
    
    return Na_entry_partial * correction_factor
end


# # Example
# V_rest = -65.0
# V_after_spike = -55.0  # Haven't fully repolarized

# Na_from_rest = sodium_entry_per_spike(V_rest)
# Na_from_elevated = sodium_entry_per_spike(V_after_spike)

# println("Na+ entry per spike:")
# println("  From rest: $(round(Na_from_rest, digits=2))x baseline")
# println("  From -55mV: $(round(Na_from_elevated, digits=2))x baseline")
# println("  Increase: $(round(100*(Na_from_elevated/Na_from_rest - 1), digits=1))%")

"""
Intracellular sodium accumulation at high firing rates
"""
function sodium_accumulation(firing_rate_hz, duration_s=1.0;
                            Na_per_spike=10e-15,  # moles
                            cell_volume=1e-15,     # liters (1 pL)
                            pump_rate=1e-14)       # moles/s
    
    # Na+ influx
    Na_influx_rate = Na_per_spike * firing_rate_hz  # moles/s
    
    # Na+ efflux (pump)
    # Pump rate increases with [Na+]ᵢ but saturates
    
    # Steady-state [Na+]ᵢ when influx = efflux
    # Simplified: Na_ss = Na_rest + (influx / pump_efficiency)
    
    Na_rest = 10e-3  # M (10 mM resting)
    
    # At high rates, pump can't keep up
    pump_max = pump_rate / cell_volume  # M/s
    Na_accumulation = (Na_influx_rate / cell_volume) / pump_max
    
    Na_steady = Na_rest * (1 + Na_accumulation)
    
    return (
        Na_rest = Na_rest * 1e3,  # Convert to mM
        Na_steady = Na_steady * 1e3,
        fold_increase = Na_steady / Na_rest
    )
end

# # Compare low vs high rates
# for rate in [10, 50, 100, 200]
#     result = sodium_accumulation(rate)
#     println("At $(rate) Hz:")
#     println("  [Na+]ᵢ: $(round(result.Na_steady, digits=1)) mM")
#     println("  Increase: $(round(result.fold_increase, digits=2))x")
# end

"""
Intracellular sodium accumulation at high firing rates
"""
function sodium_accumulation(firing_rate_hz, duration_s=1.0;
                            Na_per_spike=10e-15,  # moles
                            cell_volume=1e-15,     # liters (1 pL)
                            pump_rate=1e-14)       # moles/s
    
    # Na+ influx
    Na_influx_rate = Na_per_spike * firing_rate_hz  # moles/s
    
    # Na+ efflux (pump)
    # Pump rate increases with [Na+]ᵢ but saturates
    
    # Steady-state [Na+]ᵢ when influx = efflux
    # Simplified: Na_ss = Na_rest + (influx / pump_efficiency)
    
    Na_rest = 10e-3  # M (10 mM resting)
    
    # At high rates, pump can't keep up
    pump_max = pump_rate / cell_volume  # M/s
    Na_accumulation = (Na_influx_rate / cell_volume) / pump_max
    
    Na_steady = Na_rest * (1 + Na_accumulation)
    
    return (
        Na_rest = Na_rest * 1e3,  # Convert to mM
        Na_steady = Na_steady * 1e3,
        fold_increase = Na_steady / Na_rest
    )
end

# # Compare low vs high rates
# for rate in [10, 50, 100, 200]
#     result = sodium_accumulation(rate)
#     println("At $(rate) Hz:")
#     println("  [Na+]ᵢ: $(round(result.Na_steady, digits=1)) mM")
#     println("  Increase: $(round(result.fold_increase, digits=2))x")
# end

"""
Na/K pump efficiency as function of intracellular [Na+]
"""
function pump_efficiency(Na_internal_mM, Na_rest_mM=10.0)
    # Pump has Michaelis-Menten kinetics
    # But at high [Na+]ᵢ, approaches saturation
    
    Km_Na = 20.0  # mM (half-saturation)
    
    # Pump rate: V = Vmax * [Na]ᵢ / (Km + [Na]ᵢ)
    rate_rest = Na_rest_mM / (Km_Na + Na_rest_mM)
    rate_elevated = Na_internal_mM / (Km_Na + Na_internal_mM)
    
    # Efficiency: how much faster than resting
    relative_rate = rate_elevated / rate_rest
    
    # But ATP cost per Na+ increases as pump works harder
    # Empirically, ~1.5-2x more ATP per Na+ at high [Na+]
    atp_cost_factor = 1.0 + 0.5 * (Na_internal_mM - Na_rest_mM) / Na_rest_mM
    
    return (
        relative_rate = relative_rate,
        atp_per_na = atp_cost_factor,
        net_efficiency = relative_rate / atp_cost_factor
    )
end

function show_pump_efficiency()


    Na_range = 10:0.5:40  # mM
    efficiencies = [pump_efficiency(Na) for Na in Na_range]

    fig = Figure(size=(1200, 400))

    ax1 = Axis(fig[1, 1], xlabel="[Na+]ᵢ (mM)", ylabel="Pump Rate (relative)",
            title="Pump Rate vs [Na+]")
    lines!(ax1, Na_range, [e.relative_rate for e in efficiencies], linewidth=3)

    ax2 = Axis(fig[1, 2], xlabel="[Na+]ᵢ (mM)", ylabel="ATP per Na+ (relative)",
            title="Energy Cost per Ion")
    lines!(ax2, Na_range, [e.atp_per_na for e in efficiencies], linewidth=3)

    ax3 = Axis(fig[1, 3], xlabel="[Na+]ᵢ (mM)", ylabel="Net Efficiency",
            title="Net Pump Efficiency")
    lines!(ax3, Na_range, [e.net_efficiency for e in efficiencies], linewidth=3)
    hlines!(ax3, [1.0], linestyle=:dash, color=:red, label="Baseline")

    fig

end

using Distributions

"""
Energy consumption with high-frequency corrections
"""
function energy_with_high_freq_correction(firing_rate_hz;
                                         E_baseline=5e9,
                                         E_spike_base=1e7)
    
    # Baseline energy
    E_rest = E_baseline
    
    # Spike energy with corrections
    if firing_rate_hz < 50
        # Low rate: linear
        correction_factor = 1.0
    elseif firing_rate_hz < 100
        # Moderate rate: slight increase
        correction_factor = 1.0 + 0.3 * (firing_rate_hz - 50) / 50
    else
        # High rate: substantial increase
        # Factor increases as ~1 + 0.5*(f/100)^1.5
        excess_rate = (firing_rate_hz - 100) / 100
        correction_factor = 1.3 + 0.5 * excess_rate^1.5
    end
    
    E_spiking = E_spike_base * firing_rate_hz * correction_factor
    
    # Synaptic costs (also increase)
    E_synaptic_base = 1.5e8 * firing_rate_hz
    
    # At very high rates, synaptic machinery also struggles
    if firing_rate_hz > 100
        synaptic_factor = 1.0 + 0.3 * (firing_rate_hz - 100) / 100
        E_synaptic = E_synaptic_base * synaptic_factor
    else
        E_synaptic = E_synaptic_base
    end
    
    # Additional cost: heat dissipation, metabolic stress
    if firing_rate_hz > 150
        E_stress = 1e9 * (firing_rate_hz - 150) / 50
    else
        E_stress = 0.0
    end
    
    return (
        total = E_rest + E_spiking + E_synaptic + E_stress,
        baseline = E_rest,
        spiking = E_spiking,
        synaptic = E_synaptic,
        stress = E_stress,
        correction_factor = correction_factor
    )
end

function compare_linear_nonlinear_power_consumption()
    # Compare linear vs corrected
    rates = 0:5:300
    E_linear = [5e9 + 1.6e8 * r for r in rates]
    E_corrected = [energy_with_high_freq_correction(r).total for r in rates]

    fig = Figure(size=(1200, 800))

    ax1 = Axis(fig[1, 1],
            xlabel="Firing Rate (Hz)",
            ylabel="Energy (billion ATP/s)",
            title="Linear vs Corrected Energy Model")

    lines!(ax1, rates, E_linear ./ 1e9, linewidth=3, 
        label="Linear (incorrect)", linestyle=:dash)
    lines!(ax1, rates, E_corrected ./ 1e9, linewidth=3,
        label="With high-rate corrections", color=:red)

    vlines!(ax1, [100], linestyle=:dot, label="100 Hz threshold")

    axislegend(ax1, position=:lt)

    # Correction factor
    ax2 = Axis(fig[1, 2],
            xlabel="Firing Rate (Hz)",
            ylabel="Energy per Spike (relative)",
            title="Increased Cost per Spike at High Rates")

    corrections = [energy_with_high_freq_correction(r).correction_factor 
                for r in rates]
    lines!(ax2, rates, corrections, linewidth=3, color=:red)
    hlines!(ax2, [1.0], linestyle=:dash, color=:black, label="Baseline")

    axislegend(ax2)

    # Energy per spike
    ax3 = Axis(fig[2, 1:2],
            xlabel="Firing Rate (Hz)",
            ylabel="Total Energy / Firing Rate",
            title="Average Energy per Spike (includes all costs)")

    E_per_spike = E_corrected ./ (rates .+ 0.1)
    lines!(ax3, rates[2:end], E_per_spike[2:end] ./ 1e6, linewidth=3)

    fig

end

"""
Energy cost increases due to incomplete recovery from inactivation
"""
function refractory_cost(isi_ms)  # Inter-spike interval
    # Absolute refractory period: ~1-2 ms
    # Relative refractory period: ~5-10 ms
    
    if isi_ms < 2
        # Impossible - neuron physically can't spike
        return Inf
    elseif isi_ms < 5
        # Absolute → relative: need more current to overcome inactivation
        # More Na+ channels need to recover
        extra_cost = 2.0 - 0.2 * isi_ms  # 2x cost at 2ms, 1x at 5ms
        return extra_cost
    elseif isi_ms < 20
        # Relative refractory: partial recovery
        extra_cost = 1.0 + 0.5 * exp(-(isi_ms - 5) / 5)
        return extra_cost
    else
        # Full recovery
        return 1.0
    end
end


function show_refractory_costs()
    # Convert firing rate to ISI
    firing_rates = [10, 50, 100, 200, 500]
    isis = 1000 ./ firing_rates  # ms

    println("Refractory period effects:")
    for (rate, isi) in zip(firing_rates, isis)
        cost = refractory_cost(isi)
        if isfinite(cost)
            println("  $(rate) Hz (ISI=$(round(isi, digits=1))ms): $(round(cost, digits=2))x cost")
        else
            println("  $(rate) Hz (ISI=$(round(isi, digits=1))ms): IMPOSSIBLE")
        end
    end

end

"""
Calcium accumulation at high firing rates
Affects both energy and neurotransmitter release
"""
function calcium_effects(firing_rate_hz)
    # Ca2+ enters with each spike
    # At high rates, [Ca2+]ᵢ accumulates
    
    Ca_per_spike = 1.0  # Arbitrary units
    tau_clearance = 100e-3  # 100 ms clearance time constant
    
    # Steady-state [Ca2+]ᵢ
    # dCa/dt = influx - Ca/tau = 0
    # Ca_ss = influx * tau = (Ca_per_spike * rate) * tau
    
    Ca_steady = Ca_per_spike * firing_rate_hz * tau_clearance
    
    # Energy cost of Ca2+ pumps
    # Each Ca2+ costs ~1 ATP (via Ca-ATPase or Na-Ca exchanger)
    ATP_per_second = Ca_per_spike * firing_rate_hz
    
    # At high [Ca2+], pumps less efficient
    if Ca_steady > 0.5  # Threshold for pump saturation
        pump_efficiency = 1.0 / (1.0 + Ca_steady)
        ATP_corrected = ATP_per_second / pump_efficiency
    else
        ATP_corrected = ATP_per_second
    end
    
    return (
        Ca_steady = Ca_steady,
        ATP_for_Ca = ATP_corrected,
        pump_efficiency = Ca_steady > 0.5 ? 1.0/(1.0 + Ca_steady) : 1.0
    )
end

function show_calcium_costs()

println("\nCalcium handling costs:")
for rate in [10, 50, 100, 200]
    result = calcium_effects(rate)
    println("At $(rate) Hz:")
    println("  [Ca2+] steady-state: $(round(result.Ca_steady, digits=2)) AU")
    println("  Pump efficiency: $(round(result.pump_efficiency, digits=2))")
end

end

"""
Synaptic vesicle depletion at high firing rates
"""
function vesicle_depletion_cost(firing_rate_hz, n_synapses=10000)
    # Readily releasable pool (RRP): ~10 vesicles per synapse
    RRP_size = 10
    refill_rate = 20  # vesicles/s per synapse (slow!)
    
    # Demand
    vesicles_per_sec = firing_rate_hz * n_synapses
    
    # Supply
    vesicles_available = RRP_size * n_synapses  # Total pool
    refill_per_sec = refill_rate * n_synapses
    
    # If demand > refill rate, deplete pool
    if firing_rate_hz > refill_rate / 1.0  # Simplification
        # Need to mobilize reserve pool - costs more energy
        depletion_factor = firing_rate_hz / refill_rate
        
        # Energy cost increases: need to mobilize reserve pool
        # and accelerate vesicle trafficking
        extra_energy_factor = 1.0 + 0.5 * (depletion_factor - 1.0)
    else
        extra_energy_factor = 1.0
    end
    
    return (
        depletion_factor = depletion_factor,
        extra_cost = extra_energy_factor,
        sustainable = firing_rate_hz < refill_rate
    )
end

function show_depletion_costs()
println("\nVesicle depletion effects:")
for rate in [10, 30, 50, 100, 200]
    result = vesicle_depletion_cost(rate)
    println("At $(rate) Hz:")
    println("  Sustainable: $(result.sustainable)")
    println("  Extra energy: $(round(result.extra_cost, digits=2))x")
end
end

"""
Comprehensive energy model including all high-frequency effects
"""
function comprehensive_energy_model(firing_rate_hz)
    # Base costs
    E_baseline = 5e9
    E_spike_base = 1e7
    E_synapse_base = 1.6e8
    
    # 1. Incomplete repolarization
    isi_ms = 1000.0 / max(firing_rate_hz, 1.0)
    repol_factor = refractory_cost(isi_ms)
    
    # 2. Na+ accumulation and pump efficiency
    Na_result = sodium_accumulation(firing_rate_hz)
    pump_result = pump_efficiency(Na_result.Na_steady)
    
    # 3. Ca2+ handling
    Ca_result = calcium_effects(firing_rate_hz)
    
    # 4. Vesicle depletion
    vesicle_result = vesicle_depletion_cost(firing_rate_hz)
    
    # Combined factors
    if !isfinite(repol_factor)
        # Physically impossible rate
        return (total=Inf, breakdown="Rate too high - violates refractory period")
    end
    
    spike_cost_factor = repol_factor * pump_result.atp_per_na
    synaptic_cost_factor = vesicle_result.extra_cost
    
    # Total energy
    E_spiking = E_spike_base * firing_rate_hz * spike_cost_factor
    E_synaptic = E_synapse_base * firing_rate_hz * synaptic_cost_factor
    E_calcium = Ca_result.ATP_for_Ca * 1e6  # Convert to ATP/s
    
    E_total = E_baseline + E_spiking + E_synaptic + E_calcium
    
    return (
        total = E_total,
        baseline = E_baseline,
        spiking = E_spiking,
        synaptic = E_synaptic,
        calcium = E_calcium,
        spike_factor = spike_cost_factor,
        synaptic_factor = synaptic_cost_factor,
        sustainable = vesicle_result.sustainable,
        Na_accumulation = Na_result.fold_increase,
        pump_efficiency = pump_result.net_efficiency
    )
end

function show_power_consumption_full()

# Test across range
rates = [1, 10, 50, 100, 150, 200, 300, 500]

println("="^70)
println("COMPREHENSIVE ENERGY MODEL")
println("="^70)

for rate in rates
    result = comprehensive_energy_model(rate)
    
    if !isfinite(result.total)
        println("\n$(rate) Hz: $(result.breakdown)")
        continue
    end
    
    println("\n$(rate) Hz:")
    println("  Total energy: $(round(result.total/1e9, digits=1)) billion ATP/s")
    println("  Spike cost factor: $(round(result.spike_factor, digits=2))x")
    println("  Synaptic cost factor: $(round(result.synaptic_factor, digits=2))x")
    println("  Na+ accumulation: $(round(result.Na_accumulation, digits=2))x baseline")
    println("  Pump efficiency: $(round(result.pump_efficiency, digits=2))")
    println("  Sustainable: $(result.sustainable)")
end

end