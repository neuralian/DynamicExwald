using Random

"""
    voss_pink_noise(n_samples, n_generators=12)

Generates `n_samples` of pink noise using the Voss-McCartney algorithm.
`n_generators` (columns) determines the depth; 12-16 is usually plenty 
for audio or standard simulations.
"""
function voss_pink_noise(n_samples, n_generators=12)
    # Matrix to hold the current value of each generator
    # We use a matrix to track values across 'n_generators' levels
    generators = randn(n_generators)
    current_sum = sum(generators)
    
    out = Vector{Float64}(undef, n_samples)
    
    # Track the number of trailing zeros to decide which generator to update
    # This ensures generator i updates every 2^i steps
    for i in 1:n_samples
        # Count trailing zeros in the binary representation of the index
        # trailing_zeros(i) returns 0 for odd, 1 for multiples of 2, 2 for multiples of 4...
        idx = trailing_zeros(i) + 1
        
        if idx <= n_generators
            old_val = generators[idx]
            generators[idx] = randn()
            current_sum += generators[idx] - old_val
        end
        
        out[i] = current_sum
    end
    
    # Normalize to mean 0 and standard deviation 1 (optional)
    out .-= mean(out)
    out ./= std(out)
    
    return out
end

# --- Visualization and Verification ---
using GLMakie, Statistics, DSP

n = 2^16
pink = voss_pink_noise(n, 1)

# Compute Power Spectral Density (PSD)
# Pink noise should show a -10dB/decade (1/f) slope
p = periodogram(pink)

fig = Figure(size = (800, 600))
ax_time = Axis(fig[1, 1], title="Time Series (Pink Noise)", xlabel="Samples", ylabel="Amplitude")
ax_psd = Axis(fig[2, 1], title="Power Spectral Density (Log-Log)", 
              xlabel="Frequency", ylabel="Power",
              xscale=log10, yscale=log10)

lines!(ax_time, pink[1:1000], color=:blue)
lines!(ax_psd, p.freq[2:end], p.power[2:end], color=:red)

display(fig)