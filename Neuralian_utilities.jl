# Neuralian Toolbox Utilities
# misc tools including pdfs
# MGP 2024-25


# plot spiketime vector as spikes
function splot(spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0)

    linesegments(vec([( Point2f(t, 0.0), Point2f(t, height)) for t in spiketime]),
        linewidth = lw, color = :blue)

end


function splot!(ax::Axis, spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0, color = :blue; label::String="")

    spikes = linesegments!(ax, vec([( Point2f(t, 0.0), Point2f(t, height)) for t in spiketime]),
        linewidth = lw, color = color, label = label)
    baseline = lines!(ax, [0.0, spiketime[end]], [0.0, 0.0], color = color)
    (spikes, baseline)
end

# binary Float64 (1.0 or 0.0) vector from spike times 
# nb spike times = cumsum(intervals)
function spiketimes2binary(spiketime::Vector{Float64}, dt::Float64, T::Float64=ceil(maximum(spiketime)))

    t = dt:dt:T
    binarySpike = zeros(length(t))
    #@infiltrate
    for i in 1:length(spiketime)
        if spiketime[i] <= T
            binarySpike[Int(round(spiketime[i] / dt))] = 1.0
        end
    end

    binarySpike
end

# play spike train audio 
function listen(spiketime::Vector{Float64})

    audioSampleFreq = 8192.0

    spikeAudioData = spiketimes2binary(spiketime, 1.0 / audioSampleFreq)
    PortAudioStream(0, 2; samplerate=audioSampleFreq) do stream
           write(stream, spikeAudioData)
       end
end

# write mp3 file spike train audio 
function spiketimes2mp3(spiketime::Vector{Float64}, fileName::String="spiketrain")

    audioSampleFreq = 8192.0
 
    spikeAudioData = spiketimes2binary(spiketime, 1.0 / audioSampleFreq)
    wavwrite(spikeAudioData, fileName*".wav", Fs=audioSampleFreq)
end

# Gaussian local rate (GLR) estimate, bandwidth f up to time T
function GLR(spt::Vector{Float64}, f::Float64, T::Float64=-1, dt::Float64 = 0.01)

    if T<0.0
        T = maximum(spt)
    end

    # sample times
    t = collect(0.0:dt:T)

    # Gaussian filter width for Gain = 1/sqrt(2) = -3dB
    sd = sqrt(log(2))/(2π*f)

    # sampled sum of Gaussians centred at spike times, also sample times
    return sum(pdf.(Normal(st, sd), t) for st in spt), t

end

# Gaussian local rate (GLR) estimate, cosine-modulated + rectified
#   - linear gain = 1.0 exactly at DC (0 Hz)
#   - linear gain = 1.0 exactly at the oscillation frequency f Hz
#   - kernel may go negative (interpretable as "invisible negative spikes")
#   - final rate clipped ≥ 0 (realistic non-negative firing rate)
#   - same kernel width as original GLR (graceful roll-off, no ringing)
function GLRF(spt::Vector{Float64}, f::Float64, T::Float64=-1, dt::Float64 = 0.01)

    if T < 0.0
        T = maximum(spt)
    end

    # sample times
    t = collect(0.0:dt:T)

    # base Gaussian width (exactly the same as original: -3 dB at f Hz)
    sd = sqrt(log(2)) / (2π * f)

    # frequency response of the base Gaussian at the target frequencies
    ω0 = 2π * f
    G_f  = exp(-0.5 * (sd * ω0)^2)          # exactly 1/√2
    G_2f = exp(-0.5 * (sd * 2ω0)^2)         # exactly 0.25

    # solve for α, β so that the composite filter H(0) = 1 and H(ω0) = 1
    # (derived from H(ω) = α G(ω) + (β/2)[G(ω-ω0) + G(ω+ω0)])
    denom = 0.5 * (1 + G_2f) - G_f^2
    β = (1 - G_f) / denom
    α = 1 - β * G_f

    # build modulated kernel (may be negative)
    rate = zeros(length(t))
    for st in spt
        g = pdf.(Normal(st, sd), t)
        rate .+= α .* g .+ β .* g .* cos.(ω0 .* (t .- st))
    end

    # rectifier: clip negative values (consistent with "invisible negative spikes")
    rate = max.(0.0, rate)

    return rate, t
end


# Kullback-Liebler divergence from ISI data to model
# e.g loss function for optimization to fit Exwald model to ISI data
#   KLD = (param, grad) -> sKLD(ISI, d->Exwaldpdf(param..., d) )
# see Paulin, Pullar and Hoffman (2024) Sec. 2.2.3
function sKLD(interval::Vector{Float64}, model::Function)

    N = length(interval)
    return -sum(log2.(model(interval)))/N + log2(N)
end

# Kullback-Liebler divergence from qSLIF spiking neuron model to Exwald model of spontaneous firing 
# i.e. how well does the spontaneous ISI distribution of the spiking neuron model
#      fit the specified Exwald distribution, for fitting qSLIF models to Exwald models of data.
# NB qSLIF is parameterized by IG(μ, λ) corresponding to the IG distribution in the limiting case
#    of SLIF with long membrane time constant (Wald process), plus the time constant τ and the 
#    fractional integration parameter q. qSLIFparam = [μ, λ, τ_membrane, q].   
# Loss function for optimization is:
#   KLD = (param, grad) -> qSLIF2Exwald_KLD([μ_neuron, λ_neuron, τ_neuron, q],[μ_x, λ_x, τ_x, t], N )
# NB the two sets of (μ, λ, τ) are not the same (but μ and λ should be similar for large τ_neuron)
# N is number of spikes (intervals) to generate
function qSLIF2Exwald_KLD(qSLIFparam::Vector{Float64}, Exwaldparam::Vector{Float64}, 
        N::Int64=5000, dt = DEFAULT_SIMULATION_DT)

    println(qSLIFparam[1], ", ", qSLIFparam[2], ", ", qSLIFparam[3], ", ", qSLIFparam[4])

    # spiking neuron model
    neuron, _  = make_qSLIFc_neuron(qSLIFparam...)

    # probability of interval greated than .01 crititical point + 10τ is < 1 in 2 million 
    # (assuming tail is exponential τ)
    timeout = Exwald_tcrit(Exwaldparam..., 0.01) + 10.0*Exwaldparam[3]

    # spontaneous spikes (intervals) generated by neuron model 
    ISI = zeros(Float64, N)

    t = 0.0
    previous_spiketime = 0.0
    i = 0

    while i < N
        t += dt
        Delta_t = t - previous_spiketime
        if Delta_t > timeout   # break if interval this long is impossible under Exwald model
            i = i + 1
            ISI[i] = Delta_t
            #N = i
            println("Δt = ", Delta_t )
            #break
        elseif neuron(t->0.0, t)==true   # neuron fired spontaneously at t
            i += 1 
            ISI[i] = Delta_t
            previous_spiketime = t  
        end
    end

    # this will call the vectorised fast Gauss-Legendre version of Exwald 
   # return -sum(log2.(Exwaldpdf(Exwaldparam..., ISI[1:N])))/N + log2(N)

    kld_pq, kld_qp = KLD_Exwald2ISI(ISI, Exwaldparam, n_nodes = 60, floor = 1e-10)
    return kld_pq

  #  return sKLD(ISI, u->Exwaldpdf(Exwaldparam..., u))
   #return KLD_Exwald2ISI(ISI, Exwaldparam, n_nodes = 60, floor = 1e-10)

end

# 
function Jenson_Shannon_divergence(intervals::Vector{Float64}, 
                  Exwaldparam::Vector{Float64};
                  n_nodes::Int64 = 60, floor::Float64 = 1e-10)
   
    n      = length(intervals)
    
    q_vals = Exwaldpdf(Exwaldparam..., intervals, n_nodes)
    
    # KDE with Silverman bandwidth
    h      = 1.06 * std(intervals) * n^(-0.2)
    p_vals = [mean(exp.(-(t .- intervals).^2 ./ (2h^2))) / (h * sqrt(2π))
              for t in intervals]

    # Keep only points where both densities are above floor
    # — discard tail points where numerics are unreliable
    mask   = (p_vals .> floor) .& (q_vals .> floor)
    p_vals = p_vals[mask]
    q_vals = q_vals[mask]

    if sum(mask) < 10
        @warn "Too few reliable points in JSD evaluation — check parameters"
        return 1.0    # return maximum JSD as penalty
    end

    # Normalise to ensure they integrate to 1 over retained points
    # (masking breaks normalisation slightly)
    p_vals ./= sum(p_vals)
    q_vals ./= sum(q_vals)

    m_vals  = (p_vals .+ q_vals) ./ 2.0

    # Clip log ratios to [-20, 0] — JSD terms are always ≤ 0 before negation,
    # and -20 bits is effectively zero contribution
    kld_pm  = mean(clamp.(log2.(p_vals ./ m_vals), -20.0, 0.0))
    kld_qm  = mean(clamp.(log2.(q_vals ./ m_vals), -20.0, 0.0))

    #kld_pq = mean(clamp.(log2.(q_vals ./ m_vals), -20.0, 0.0))


    jsd     = -(kld_pm + kld_qm) / 2.0    # negate because log(p/m) ≤ 0 always

    # Clamp final result to [0, 1] as a safety net
    return clamp(jsd, 0.0, 1.0)
end



# 
function KLD_Exwald2ISI(intervals::Vector{Float64}, 
                  Exwaldparam::Vector{Float64};
                  n_nodes::Int64 = 60, floor::Float64 = 1e-10)
   
    n      = length(intervals)
    
    q_vals = Exwaldpdf(Exwaldparam..., intervals, n_nodes)
    
    # Kernel Density estimate of ISI distribution with Silverman bandwidth
    h      = 10.0 * std(intervals) * n^(-0.2) #1.06 * std(intervals) * n^(-0.2)
    p_vals = [mean(exp.(-(t .- intervals).^2 ./ (2h^2))) / (h * sqrt(2π))
              for t in intervals]

    # Keep only points where both densities are above floor
    # — discard tail points where numerics are unreliable
    mask   = (p_vals .> floor) .& (q_vals .> floor)
    p_vals = p_vals[mask]
    q_vals = q_vals[mask]

    if sum(mask) < 10
        @warn "Too few reliable points in KLD evaluation — check parameters"
        return 100.0, 100.0    # return maximum JSD as penalty
    end

    # Normalise to ensure they integrate to 1 over retained points
    # (masking breaks normalisation slightly)
    p_vals ./= sum(p_vals)
    q_vals ./= sum(q_vals)

    # -20 bits is effectively zero contribution
    kld_pq  = -mean(clamp.(log2.(p_vals ./ q_vals), -20.0, 0.0))
    kld_qp  = -mean(clamp.(log2.(q_vals ./ p_vals), -20.0, 0.0))

    return kld_pq, kld_qp
end

# # Kullback-Liebler divergence from ISI data to model
# # see Paulin, Pullar and Hoffman (2024) Sec. 2.2.3
# function sKLD(interval::Vector{Float64}, model::Function)

#     N = length(interval)
#     return -sum(log2.([model(interval[i]) for i in 1:N]))/N + log2(N)
# end


# Kullback-Leibler divergence from p(x) to q(x) 
# where p, q are relative frequency histograms or sampled pdfs with binwidth bw
function KLD( p::Vector{Float64}, q::Vector{Float64}, bw::Float64)

    @assert length(p)==length(q)  "p and q must be the same length"
    @assert all(p .>= 0.0)                      "p must be non-negative"
 #   @assert all(q .>= 0.0)                       "q must be positive"
    @assert isapprox(sum(p)*bw, 1.0, atol = 1.0e-6)  "p*bw must sum to 1"
  #  @assert isapprox(sum(q)*bw, 1.0, atol = 1.0e-6)  "q*bw must sum to 1"

#    if isapprox(isapprox(sum(q)*bw, 1.0, atol = 1.0e-6))==false || any(q.<0.0)
#     @infiltrate
#    end

    D = 0.0
    for i in 1:length(p)
        if p[i] > 0.0  && q[i] > 0.0
            D += p[i]*log(p[i]/q[i])
        end
    end

    return D

end

# entropy H and channel capacity of Exwald distribution discretized at specified timescale, in bits
# nb using jaynes limiting density of points
function Exwald_entropy(EXWparam::Tuple{Float64, Float64, Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    (μ, λ, τ) = EXWparam
    s = sqrt(μ^2/λ + τ^2)   # s.d. of density
    t = dt:dt:(μ+6.0*s)     # evaluate to 6 sigma beyond mean (way too far, don't care)
    
    S = 0.0
    N = 0
    for i = 1:length(t)
        f = Exwaldpdf(μ, λ, τ, t[i])*dt
        if f > 0.0
            S += -f*log2(f)
            N += 1
        end
    end

    S += log2(N)

    # S = entropy 
    # C = spontaneous channel capacity = bits/spike * spikes/second = bits/second 
    # P = power consumption assuming constant watts/ATP per spike
    P = 1.0/(μ+τ)
    return S, S*P, P

end

# information (bits) in Exwald spikes relative to Poisson spikes at the same rate
# 
function KLD_Exwald_to_Exponential(EXWparam::Tuple{Float64, Float64, Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    (μ, λ, τ) = EXWparam
    m = μ+τ  # mean interval 
    s = sqrt(μ^2/λ + τ^2)   # s.d. of density
    t = dt:dt:(μ+6.0*s)     # evaluate to 6 sigma beyond mean (way too far, don't care)
    
    S = 0.0
    for i = 1:length(t)
        p = Exwaldpdf(μ, λ, τ, t[i])*dt    # Exwald probability in bin
        q = exp(-t[i]/m)/m*dt              # Exponential probability in bin
        if p > 0.0 && q > 0.0
            S += p*log2(p/q)
        end
    end

    return S


end

# Faddeeva w function
function Faddeeva_w(x::Union{Float64, ComplexF64})
    erfcx(-im*x)
end



# Exwald pdf by convolution of Wald and Exponential pdf
function Exwaldpdf_byconvolution(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64})

    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau
    X = imfilter(W, reflect(P))
    X = X / sum(X) / mean(diff(t)) # renormalize (W & P are not normalized because of discrete approx)
    #@infiltrate
end

# Exwald pdf at t (From Schwarz (2002) DOI: 10.1081/STA-120017215)
# turns out that for realistic neuron models Schwarz's formula entails cancellation 
# of very large exponentials, causing intractable numerical error. 
# So it's rubbish for this application. Code retained for historical interest.
function Exwaldpdf_Schwarz(mu::Float64, lambda::Float64, tau::Float64, t::Float64)

    @assert t >= 0.0         "Exwald undefined for t < 0.0"

    # use Schwarz (2002) notation to make it easier to check formulas
    # drift-diffusion process with drift rate μ, noise s.d. sigma, barrier height L = 1.0
    μ, sigma, L = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)
    # λ = Poisson rate
    λ = 1.0/tau
    # s = evaluation point
    s = t

    # case μ² ≥ 2λ sigma^2 (Schwarz Section 3.1 p2118)
    disc = μ^2 - 2.0*λ*sigma^2  

    # Special cases

    # easy
    if t == 0.0
        return 0.0
    end

    # if tau <= 0.0 return Wald
    if tau <= 0.0 
        return pdf(InverseGaussian(mu, lambda), t)
    end

    # if Wald parameters are invalid return Exponential
    if mu <= 0.0  || lambda <= 0.0  #|| L*μ/sigma^2 > 700.
        return pdf(Exponential(tau), t)
    end

    # case  μ² ≥ 2λ sigma^2 (Schwarz Section 3.1 p2118)
    if disc > 0.0
        k = sqrt(disc) 
       # println(μ-k, ", ", μ + k)

        f = λ*exp(-λ*s)*exp(L*(μ-k)/sigma^2)*cdf(Normal(0.0, 1.0), (k*s-L)/(sigma*sqrt(s))) 
           + λ*exp(-λ*s)exp(L*(μ+k)/sigma^2)*cdf(Normal(0.0, 1.0), -(k*s+L)/(sigma*sqrt(s))) 
                              
    else # k2 < 0
        k = sqrt(-disc)
        f = λ*exp(-(L-μ*s)^2/(2.0*sigma^2*s))*real(Faddeeva_w(k*sqrt(s)/(sigma*sqrt(2.0)) + im*L/(sigma*sqrt(2.0*s))))
    end

    return f
end

# Quadrature is accurate, so this is the official (default) function for computing Exwald at t.
function Exwaldpdf( mu::Float64, lambda::Float64, tau::Float64, t::Float64)
    t <= 0.0 && return 0.0
    ig(u)  = u > 0 ? sqrt(lambda/(2π*u^3)) * exp(-lambda*(u-mu)^2/(2*mu^2*u)) : 0.0
    ex(s)  = s > 0 ? exp(-s/tau)/tau : 0.0
    val, _ = quadgk(u -> ig(u) * ex(t - u), 1e-10, t, rtol=1e-8)
    return val
end

# This is the simple way to compute Exwald at a set of points t-1, t_2, ..., t_n
# But there is a better, faster way, below ...
Exwaldpdf_vec( mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}) = 
    [Exwaldpdf( mu, lambda, tau, s) for s in t]




#  Exwaldpdf for vector t, using Gauss-Legendre quadrature.
# 
#   Exwaldpdf(t) = ∫₀ᵗ IG(u; μ,λ) · Exp(t-u; τ) du   (defn).
#
#  map u = t·v (v ∈ [0,1]) then 
#
#     Exwaldpdf(t) = t · ∫₀¹ f_IG(t·v) · f_Exp(t·(1-v)) dv
#
# This has the same Gauss-Legendre nodes v_k ∈ [0,1] for every t,
# so the vector of points can be evaluated in a single sweep.
# Edited by MGP based on code by Claude Sonnet 4.6, 27 March 2026
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, 
                    n_nodes::Int = 60 )

    # Gauss-Legendre nodes and weights on [0,1]
    nodes, weights = gausslegendre(n_nodes)
    # nodes are on [-1,1], map to [0,1]
    v_k = (nodes .+ 1.0) ./ 2.0
    w_k = weights ./ 2.0           # Jacobian of [−1,1]→[0,1]

    n   = length(t)
    pdf = zeros(Float64, n)

    # Precompute IG and Exp on the 2D grid (n_nodes × n_t) at once
    # u_ik = t[i] * v_k  — matrix of quadrature points
    # Each column is one t value, each row is one quadrature node

    # t is (n,), v_k is (n_nodes,)
    # U[k,i] = t[i] * v_k[k]
    U = v_k * t'               # (n_nodes × n) matrix

    # IG at u = U[k,i]
    function ig(u)
        u <= 0.0 && return 0.0
        sqrt(lambda / (2π * u^3)) * exp(-lambda * (u - mu)^2 / (2 * mu^2 * u))
    end

    # Exp at s = t - u = t[i]*(1 - v_k[k])
    S   = (1.0 .- v_k) * t'   # (n_nodes × n)

    IG  = ig.(U)                # (n_nodes × n)  — broadcasted
    EX  = exp.(-S ./ tau) ./ tau

    # Integrand values at each node/sample pair
    INTEG = IG .* EX            # (n_nodes × n)

    # Quadrature: f(tᵢ) = tᵢ · Σₖ w_k · INTEG[k,i]
    # (the tᵢ factor is the Jacobian of u = tᵢ·v)
    f_t = t .* (w_k' * INTEG)' |> vec

    return f_t
end

"""
    Exwald_tcrit(mu, lambda, tau, p; n_nodes=60)

Find t_crit such that Pr(X > t_crit) = p for X ~ Exwald(mu, lambda, tau).

i.e. the (1-p) quantile of the Exwald distribution.

# Arguments
- `mu`, `lambda`, `tau` : Exwald parameters
- `p`                   : Upper tail probability (0 < p < 1)
- `n_nodes`             : Gauss-Legendre nodes for PDF quadrature (default 60)
"""
function Exwald_tcrit( mu::Float64,  lambda::Float64, tau::Float64, p::Float64;
    n_nodes::Int = 60 )

    @assert 0.0 < p < 1.0 "p must be in (0, 1)"

    # CDF at t via trapezoidal integration of Exwaldpdf
    function cdf(t)
        t <= 0.0 && return 0.0
        ts = collect(range(1e-10, t, length=200))
        ps = Exwaldpdf(mu, lambda, tau, ts, n_nodes)
        dt = diff(ts)
        return sum((ps[1:end-1] .+ ps[2:end]) ./ 2.0 .* dt)
    end

    # Bracket
    mean_ew = mu + tau
    std_ew  = sqrt(mu^3/lambda + tau^2)
    t_lo    = 1e-10
    t_hi    = mean_ew + 6.0 * std_ew

    # Extend upper bracket if needed (e.g. very small p)
    while cdf(t_hi) < (1.0 - p)
        t_hi *= 2.0
    end

    # Find root using Brent's method from Roots.jl
    t_crit = find_zero(t -> cdf(t) - (1.0 - p), (t_lo, t_hi), Roots.Brent())

    return t_crit
end



"""
    exwald_loglikelihood(ts; mu, lambda, tau, n_nodes=50)

Log-likelihood of observed ISIs ts under Exwald(mu, lambda, tau).
Suitable for use with Optim.jl for parameter fitting.
"""
# function exwald_loglikelihood(
#     ts::Vector{Float64};
#     mu::Float64,
#     lambda::Float64,
#     tau::Float64,
#     n_nodes::Int = 50
# )
#     pdfs = exwald_pdf_vec(ts; mu=mu, lambda=lambda, tau=tau, n_nodes=n_nodes)
#     # Guard against log(0) from numerical issues
#     return sum(log.(max.(pdfs, 1e-300)))
# end

# Exwald pdf parameterized by cv
function Exwaldpdf_cv(mu::Float64, cv::Float64, tau::Float64, t::Float64)

   Exwaldpdf(mu, mu/cv^2, tau, t)

end

# Exwald pdf parameterized by cv
function Exwaldpdf_cv(mu::Float64, cv::Float64, tau::Float64, t::Vector{Float64}, P::Bool=false)

   Exwaldpdf(mu, mu/cv^2, tau, t, P)

end



# # Exwald pdf at vector of times
# # renormalized if renorm==true
# function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, renorm::Bool=true)

#     p = [Exwaldpdf(mu, lambda, tau, s) for s in t]
#     if renorm
#         p[findall(p.<0.0)] .=0.0  
#         p /= (sum(p)*(t[2] - t[1])) # normalize as density
#     end

#     return p
# end

# # Exwald pdf at vector of times
# function scaled_Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, s::Float64)

#     [Exwaldpdf(mu, lambda, s * tau, q) for q in t]

# end


function lambertWNormal_sample(d::Float64= 0.1)

    z = randn()[]
    return w = z*exp(0.5*d*z^2)

end




# Function to compute Oustaloup approximation parameters
function oustaloup_zeros_poles(alpha::Float64, N::Int, wb::Float64, wh::Float64)
    M = 2 * N + 1
    poles = Float64[]
    zeros = Float64[]
    for k = -N:N
        # Pole frequency
        exp_p = (k + N + 0.5 * (1 + alpha)) / M
        wk = wb * (wh / wb)^exp_p
        push!(poles, wk)
        
        # Zero frequency
        exp_z = (k + N + 0.5 * (1 - alpha)) / M
        wkp = wb * (wh / wb)^exp_z
        push!(zeros, wkp)
    end

    # Normalization constant K
    K = (wh / wb)^(-alpha)
    for i in 1:M
        K *= poles[i] / zeros[i]
    end

    K *=wh^alpha
    
    return K, poles, zeros
end

# Function to compute residues for partial fraction decomposition
# nb xeros for zeros because zeros is a reserved word in Julia
function oustaloup_residues(K::Float64, poles::Vector{Float64}, xeros::Vector{Float64})
    M = length(poles)
    residues = zeros(Float64, M)
    p_locations = [-p for p in poles]  # Actual pole locations s = -ω_k
    
    for m in 1:M
        pm = p_locations[m]
        # num(pm) = K * ∏ (pm + z_k for all k)
        num_pm = K
        for zk in xeros
            num_pm *= (pm + zk)
        end
        
        # den'(pm) = ∏_{j≠m} (pm + poles[j])
        den_prime_pm = 1.0
        for j in 1:M
            if j != m
                den_prime_pm *= (pm + poles[j])
            end
        end
        
        residues[m] = num_pm / den_prime_pm
    end
    
    return residues, p_locations  # p_locations are the -ω_k, but for dynamics we use +ω_k = -p_locations[m]
end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
# selected interval is the interval containing the phase point
# (ie first interval that ends after or at the phase point)
function intervalPhase_interval(spiketime::Vector{Float64}, phase::Float64, freq::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    if phase < 0.0
        phase = 360.0 + phase
    end
    T = maximum(spiketime)
    wavelength = 1.0 / freq
    cycles = Int(floor(T / wavelength))
    sampleTime = wavelength * (phase/360.0.+0:(cycles-1))
    samnpleTime = sampleTime[findall(sampleTime .< T)[1:(end-1)]]
    interval = zeros(length(sampleTime))

    # #@infiltrate
    # # interval length at sample times. If sample time is a spike time then get next interval
    if sampleTime[1] <= spiketime[1]  # special case: first sample time is before first spike
        interval[1] = spiketime[1]
        i0 = 2
    else
        i0 = 1
    end
    for i in i0:length(sampleTime)
        iEndInterval = findfirst(spiketime .>= sampleTime[i]) 
        interval[i] = spiketime[iEndInterval] - spiketime[iEndInterval-1]
    end

    return interval

end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
# Boolean endAtClosestSpike determined whether the selected interval at a given phase angle
#     is the interval containing the phase point (Default, endAtClosestSpike==false)
#     or the interval that ends at the spiketime closest to the phase point (endAtClosestSpike==true) 
function intervalPhase(spiketime::Vector{Float64}, phase::Float64, freq::Float64, 
    endAtClosestSpike::Bool=false,  dt::Float64=DEFAULT_SIMULATION_DT)

    if phase < 0.0
        phase = 360.0 + phase
    end
    T = maximum(spiketime)
    wavelength = 1.0 / freq
    cycles = Int(floor(T / wavelength))
    sampleTime = wavelength * (phase/360.0.+0:(cycles-1))
    sampleTime = sampleTime[findall(sampleTime .< T)[1:(end-1)]]
    interval = zeros(length(sampleTime))

    # #@infiltrate
    # # interval length at sample times. If sample time is a spike time then get next interval
    if sampleTime[1] <= spiketime[1]  # special case: first sample time is before first spike
        interval[1] = spiketime[1]
        i0 = 2
    else
        i0 = 1
    end
    for i in i0:length(sampleTime)
        iEndInterval = findfirst(spiketime .>= sampleTime[i])   # index of first spike time after phase point
        if endAtClosestSpike
            # if the previous spike is closer to the phase point
            if abs(sampleTime[i]-spiketime[iEndInterval-1]) < abs(sampleTime[i]-spiketime[iEndInterval])
                iEndInterval = iEndInterval - 1    # selected interval ends at previous spike time 
            end
            if iEndInterval==1              # previous spike turns out to be the first spike
                interval[i] = spiketime[1]  # in which case the required interval is the first interval
            else
                interval[i] = spiketime[iEndInterval] - spiketime[iEndInterval-1] 
            end
        else
            interval[i] = spiketime[iEndInterval] - spiketime[iEndInterval-1]
        end
    end

    return interval

end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
# selected interval ends closest to specified phase 
# (ie if the prevous spike is closer then we pick the previous interval)
function intervalPhase_closest_spike(spiketime::Vector{Float64}, phase::Float64, freq::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    if phase < 0.0
        phase = 360.0 + phase
    end
    T = maximum(spiketime)
    wavelength = 1.0 / freq
    cycles = Int(floor(T / wavelength))
    sampleTime = wavelength * (phase/360.0.+0:(cycles-1))
    samnpleTime = sampleTime[findall(sampleTime .< T)[1:(end-1)]]
    interval = zeros(length(sampleTime))

    # #@infiltrate
    # # interval length at sample times. If sample time is a spike time then get next interval
    if sampleTime[1] <= spiketime[1]  # special case: first sample time is before first spike
        interval[1] = spiketime[1]
        i0 = 2
    else
        i0 = 1
    end
    for i in i0:length(sampleTime)

        iBefore = findlast(spiketime .<= sampleTime[i])  # index to last spike time before sampleTime[i]
        # if this spike is closer to the sample time than the next spike
        if (sampleTime[i] - spiketime[iBefore]) < (spiketime[iBefore+1] - sampleTime[i])
            # selected interval ends at spiketime[iBefore] 
            if iBefore > 1
                interval[i] = spiketime[iBefore] - spiketime[iBefore-1]
            else  # special case: closest spike is the first spike => interval is spike time
                interval[i] = spiketime[iBefore]
            end
        else
            # seleted interval ends at the following spike
            interval[i] = spiketime[iBefore+1] - spiketime[iBefore]
        end
    end

    return interval

end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
# selected interval is closest above or below interval at specified phase 
# (ie if the prevous spike is closer then we pick the previous interval)
function intervalPhase_independent(spiketime::Vector{Float64}, phase::Float64, freq::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    if phase < 0.0
        phase = 360.0 + phase
    end
    T = maximum(spiketime)
    wavelength = 1.0 / freq
    cycles = Int(floor(T / wavelength))
    sampleTime = wavelength * (phase/360.0.+0:(cycles-1))
    samnpleTime = sampleTime[findall(sampleTime .< T)[1:(end-1)]]
    interval = zeros(length(sampleTime))

    # #@infiltrate
    # # interval length at sample times. If sample time is a spike time then get next interval
    if sampleTime[1] <= spiketime[1]  # special case: first sample time is before first spike
        interval[1] = spiketime[1]
        i0 = 2
    else
        i0 = 1
    end
    for i in i0:length(sampleTime)

        iBefore = findlast(spiketime .<= sampleTime[i])  # index to last spike time before sampleTime[i]
        # if this spike is closer to the sample time than the next spike
        if (sampleTime[i] - spiketime[iBefore]) < (spiketime[iBefore+1] - sampleTime[i])
            # selected interval ends at spiketime[iBefore] 
            if iBefore > 1
                interval[i] = spiketime[iBefore] - spiketime[iBefore-1]
            else  # special case: closest spike is the first spike => interval is spike time
                interval[i] = spiketime[iBefore]
            end
        else
            # seleted interval begins at the following spike
            interval[i] = spiketime[iBefore+2] - spiketime[iBefore+1]
        end
    end

    return interval

end

function xlims(ax::Axis)
    ax.xaxis.attributes.limits[]
end

function ylims(ax::Axis)
    ax.yaxis.attributes.limits[]
end

# set bounding box for Axis in Figure
# Figure size must have been specified
function setAxisBox(ax::Axis, x0::Float64, x1::Float64, y0::Float64, y1::Float64)

    ax.scene.viewport[] = Rect(Vec(round(x0), round(x1)), Vec(round(y0), round(y1)))

end

# find nice value for single ytick on pdf plot
function niceYtick(fmax::Float64, axlim::Float64)

    # smallest power of 10 larger than xmax
    y =  10.0^ceil(log10(fmax))
    dy = y/10.0  

    # step down until within axis limit
    while y > fmax
        y -= dy 
    end

    return y 
end

# Exwald model Bode gain and phase plots 
# BLG stimulus, cross-power spectrum
function exwaldBodePlots_fromBLG(CV::Float64, Nreps::Int)
    # NS = 2^20
    # t = collect(1:NS)
    # u = randn(size(t))  # white noise
    
    # x=filt([1], [1,-0.95], u)
    
    Exwald_param = Exwald_fromCV(CV)  # Exwald model with specified CV
    
    # BLG stimulus bandwidth (Hz)
    f0 = 0.01
    f1 = 100.0
    #stimulus amplitude  
    s = 1.0 
    
    # stimlus duration
    T = 200.0
    
    # blg stimulus
    dt = 1.0e-3
    
    NN = Int(round(T/dt))
    Gain = []
    Phase = []
    Freqs = []
    iFreq = []
    
    for rep in 1:Nreps
    (tt, blg) = BLG(f0, f1, s, dt, T)
    tt = tt[2:end]
    blg = blg[2:end]   # start at t=dt not t = 0
    
    blg_fcn = t -> (t<dt) ? blg[1] : (t<=T) ? blg[Int(round(t/dt))] : blg[end]
    spiketime = Exwald_Neuron(T, Exwald_param, blg_fcn)
    
    # spike rate
    (t2, x) = GLR(spiketime, 1.0/f1, dt, 0.0, T)
    
    # # represent spike train as binary sequence 
    # x = s2b(spiketimes, dt, T)
    xspectConfig = MTCrossSpectraConfig{Float64}(2,length(tt), demean = true)
    X = DSP.allocate_output(xspectConfig)
    
    Q = mt_cross_power_spectra!(X, [blg x]', xspectConfig)
    
    # scale to Hz
    Fnyq = 1.0/(2.0*dt)
    Freqs = freq(Q)*Fnyq
    iFreq = findall(Freqs.<f1/4.0) # plot over stimulus bandwidth
    
    if rep==1
        Gain = abs.(X[1,2,2:iFreq[end]])
        Phase = -180.0/π*angle.(X[1,2,2:iFreq[end]])
    else
        Gain = Gain .+ abs.(X[1,2,2:iFreq[end]])
        Phase = Phase .- 180.0/π*angle.(X[1,2,2:iFreq[end]])    
    end
    
    end
    
    Gain = Gain/Nreps 
    Phase = Phase/Nreps 
    
    Fig = Figure(size = (800,600))
    #ax1 = Axis(Fig[1,1:2])
    ax2 = Axis(Fig[1,1], xscale = log10, yscale = log10, title = @sprintf "CV = %.2f" CV)
    ax3 = Axis(Fig[2,1], xscale = log10, yticks = [-180, -90.0, 0.0, 90, 180.0])
    ylims!(ax3, [-120.0, 120.0])
    
    
    Freqs = Freqs[2:iFreq[end]]
    
    lines!(ax2, Freqs, Gain)
    lines!(ax3, Freqs, Phase)
    
    ylims!(ax2, [0.1, 10000.])
    
    display(Fig)
    
    save("Exwald_BLG_Bode.png", Fig)
    
    Fig
end

# construct closure function to generate band-limited Gaussian noise by sum-of-cosines
#  flower, fupper:  frequency band /Hz 
#   rms:  root mean squared noise amplitude 
#  Nfreqs = number of cosines in sum, uniformly spaced in band
#
# Returns function blg(t) which returns state = (x(t), ̇x(t)) 
#                                       i.e. noise value and its rate of change at t
# 
# usage: blgStruc = (flower, fupper, rms)
#        blg = make_blg_generator(blgStruc, ...) # noise generating function with specified parameters
#        blg(t)  # noise state at time t
#  NB each blg() generates a particular noise waveform, so you can get the value and the 
# derivative at time t by calling blg() twice, i.e. x_t = blg(t)[1], dx_t =  blg(t)[2]  
function make_blg_generator(blgParams::Tuple{Float64, Float64, Float64}, Nfreqs::Int64=32)

    (flower, fupper, rms) = blgParams

    @assert flower > 0 && fupper > flower && Nfreqs>= 1  "Invalid specs for blg generator"    
    
    Δf = (fupper - flower) / Nfreqs
    fs = flower .+ (0:Nfreqs-1) * Δf        # Linearly spaced frequencies
    PSD = rms^2 / (fupper - flower)         # power spectral density
    A = sqrt(2 * PSD * Δf) * randn(Nfreqs)  # Constant scaling * Gaussian amplitudes
    phi = rand(Uniform(0, 2 * π), Nfreqs)   # Random phases
        
    function blg(t)

        x  = 0.0
        dx = 0.0
        for i in 1:Nfreqs
            x += A[i]*cos(2π*fs[i]*t + phi[i])
            dx += -2π*fs[i]*A[i]*sin(2π*fs[i]*t + phi[i])
        end

        return (x,dx) # closure function returns noise state
    end

    return blg  # enclosing function returns closure 
end

# rms power of derivative of blg noise (needed for scaling state space maps)
#  parameters = those used to construct the noise generating function
function blg_derivative_RMS(blgParams)
   
    (flower, fupper, rms) = blgParams
    sqrt( (4.0/3.0)*π^2*rms^2*(fupper^3-flower^3)/(fupper-flower) )

end

# Exwald model Bode gain and phase plots
# sinewave stimuli, vector of frequencies in Hz
# returns (freq, Gain, Phase) vectors
function exwaldGainPhase_fromSines(exwald_param::Tuple{Float64,Float64,Float64},
    freq::Vector{Float64}, amplitude::Float64)


    N = 16 # number of stimulus cycles to fit response
    dt = 1.0e-4   # time step
    (mu, lambda, tau) = exwald_param

    
    Gain = zeros(length(freq))
    Phase = zeros(length(freq))
    
    for i in 1:length(freq)

        period = 1.0/freq[i]
        Burn = Int(ceil(freq[i]))  # burn-in at least 1 second
        Ncycles = N + Burn + 1     # number of cycles to simulate (we will drop the last cycle)
        
        T = Ncycles*period   # stimulus duration

        # sinewave stimulus
        stimulus_fcn = t->sinewave([0.0, amplitude, 0.0], freq[i], t)

        # spike train response
        spiketime = Exwald_Neuron(T, exwald_param, stimulus_fcn, dt)
        

        # spike rate by Gaussian Local Rate filter
        (sampleTimes, firingRate) = GLR(spiketime, period/16.0, 0.001, 0.0, T)

        # fit sinewave to rate estimate, pest = (offset, amplitude, phase)
        (minf, pest, ret) = Fit_Sinewave_to_Spiketrain(spiketime, freq[i], dt)
 
        Gain[i] = pest[2]/amplitude
        Phase[i] = pest[3]*180.0/π   # radians to degrees
        println(i/length(freq))
        
    end

    (freq, Gain, Phase)
end

function dynamicExwaldGainPhase_fromSines(exwald_param::Tuple{Float64,Float64,Float64},
    freq::Vector{Float64}, amplitude::Float64)  

    N = 16 # number of stimulus cycles to fit response
    dt = 1.0e-4   # time step
    (mu, lambda, tau) = exwald_param

    
    Gain = zeros(length(freq))
    Phase = zeros(length(freq))
    
    for i in 1:length(freq)

        period = 1.0/freq[i]
        Burn = Int(ceil(freq[i]))  # burn-in at least 1 second
        Ncycles = N + Burn + 1     # number of cycles to simulate (we will drop the last cycle)
        
        T = Ncycles*period   # stimulus duration

        # sinewave stimulus
        stimulus_fcn = t->sinewave([0.0, amplitude, 0.0], freq[i], t)

        # spike train response
        spiketime = Exwald_Neuron(T, exwald_param, stimulus_fcn, dt)
        

        # spike rate by Gaussian Local Rate filter
        (sampleTimes, firingRate) = GLR(spiketime, period/16.0, 0.001, 0.0, T)

        # fit sinewave to rate estimate, pest = (offset, amplitude, phase)
        (minf, pest, ret) = Fit_Sinewave_to_Spiketrain(spiketime, freq[i], dt)
 
        Gain[i] = pest[2]/amplitude
        Phase[i] = pest[3]*180.0/π   # radians to degrees
        println(i/length(freq))
        
    end

    (freq, Gain, Phase)

end

# Exwald model Bode gain and phase plots 
# BLG stimulus, cross-power spectrum
function exwaldBodePlots_fromBLG(CV::Float64, Nreps::Int)
    # NS = 2^20
    # t = collect(1:NS)
    # u = randn(size(t))  # white noise
    
    # x=filt([1], [1,-0.95], u)
    
    Exwald_param = Exwald_fromCV(CV)  # Exwald model with specified CV
    
    # BLG stimulus bandwidth (Hz)
    f0 = 0.01
    f1 = 100.0
    #stimulus amplitude  
    s = 1.0 
    
    # stimlus duration
    T = 200.0
    
    # blg stimulus
    dt = 1.0e-3
    
    NN = Int(round(T/dt))
    Gain = []
    Phase = []
    Freqs = []
    iFreq = []
    
    for rep in 1:Nreps
    (tt, blg) = BLG(f0, f1, s, dt, T)
    tt = tt[2:end]
    blg = blg[2:end]   # start at t=dt not t = 0
    
    blg_fcn = t -> (t<dt) ? blg[1] : (t<=T) ? blg[Int(round(t/dt))] : blg[end]
    spiketime = Exwald_Neuron(T, Exwald_param, blg_fcn)
    
    # spike rate
    (t2, x) = GLR(spiketime, 1.0/f1, dt, 0.0, T)
    
    # # represent spike train as binary sequence 
    # x = s2b(spiketimes, dt, T)
    xspectConfig = MTCrossSpectraConfig{Float64}(2,length(tt), demean = true)
    X = DSP.allocate_output(xspectConfig)
    
    Q = mt_cross_power_spectra!(X, [blg x]', xspectConfig)
    
    # scale to Hz
    Fnyq = 1.0/(2.0*dt)
    Freqs = freq(Q)*Fnyq
    iFreq = findall(Freqs.<f1/4.0) # plot over stimulus bandwidth
    
    if rep==1
        Gain = abs.(X[1,2,2:iFreq[end]])
        Phase = -180.0/π*angle.(X[1,2,2:iFreq[end]])
    else
        Gain = Gain .+ abs.(X[1,2,2:iFreq[end]])
        Phase = Phase .- 180.0/π*angle.(X[1,2,2:iFreq[end]])    
    end
    
    end
    
    Gain = Gain/Nreps 
    Phase = Phase/Nreps 
    
    Fig = Figure(size = (800,600))
    #ax1 = Axis(Fig[1,1:2])
    ax2 = Axis(Fig[1,1], xscale = log10, yscale = log10, title = @sprintf "CV = %.2f" CV)
    ax3 = Axis(Fig[2,1], xscale = log10, yticks = [-180, -90.0, 0.0, 90, 180.0])
    ylims!(ax3, [-120.0, 120.0])
    
    
    Freqs = Freqs[2:iFreq[end]]
    
    lines!(ax2, Freqs, Gain)
    lines!(ax3, Freqs, Phase)
    
    ylims!(ax2, [0.1, 10000.])
    
    display(Fig)
    
    save("Exwald_BLG_Bode.png", Fig)
    
    Fig
end

# returns Bode gain and phase for fractional torsion pendulum - Exwald neuron model 
# as BGP = ( (Gain, GainSD), (Phase, PhaseSD), freq) 
#   Each returned variable is MxN array where M = size(freqs) & N = independent replicates
#   e.g. Gain[i,j] is the estimated gain at freq(i) on the jth replicate
# use function BodePlots(BGP) to plot.
# angularVelocity is max angular velocity in degrees per second, at all frequencies
# band is bandwith (f0, f1) Hz
# variable 'burn' specifies minimum burn-in time. We throw away the response up to the 
#   beginning of the first stimulus period after 'burn', to allow canal dynamics to 
#    reach steady-state before estimating gain and phase by fitting sines.  
# variable 'Nperiods' is the number of periods that we fit data to (independent estimates of gain & phase).
#   
function fractionalSteinhausenExwald_BodeAnalysis(q::Float64, EXWparam::Tuple{Float64,Float64,Float64},
    angularVelocity::Float64, band::Tuple{Float64, Float64}, Npts::Int64=16, 
    dt = DEFAULT_SIMULATION_DT)  

    # construct frequency vector
    freq = collect(logrange(band..., Npts))

    # Specify min burn-in time in seconds 
    burn = 10.0 

    # specify number of response cycles to fit
    Nfit = 32 

    # Extract Exwald parameters
    (mu, lambda, tau) = EXWparam

    Gain  = zeros(Float64, length(freq))
    Phase = zeros(Float64, length(freq))
    DC = zeros(Float64, length(freq))

    F = Figure()
    ax = Axis(F[1,1])
    
    for i in 1:length(freq)

        # specified stimulus amplitude amplitudeDegSec is maximum angular velocity in degrees per second
        # we need head angular acceleration in rad/s^2
        #angularAcceleration = 2π*freq[i]*angularVelocity

        period = 1.0/freq[i]
        Nburn = Int(ceil(burn/period))  # number of burn-in cycles
        #println("burn: ", Nburn)
        Ncycles = Nburn + Nfit + 1   # number of simulation cycles
                                        # including last period dropped to avoid GLR edge effect
                                                                       
        
        T = Ncycles*period   # stimulus duration

        # Sinusoidal stimulus amplitude specified as angular displacement in degrees, frequency in Hz.
        # Required input to the model is angular acceleration in radians/s^2 (check) 
        # so we convert degrees to radians (*π/180) and construct 2nd derivative of sin(2πf(t))
        # ignoring the sign change 
        w = t->sinewave([0.0, angularVelocity, 0.0], freq[i], t)

        # spike train
        spiketime = fractionalSteinhausenExwald_Neuron(q, EXWparam, w, T)

        #@infiltrate

        # spike rate by Gaussian Local Rate filter with filter width 5x spontaneous interval
        glr_dt = .001 # 1ms
        (sampleTime, firingRate) = GLR(spiketime, period/5.0, glr_dt, 0.0, T)

        # fit sine to Nfit cycles of response after burn-in
        t0 = Nburn*period
        jFit = findall(t0 .<= sampleTime .< (t0+period*Nfit))
        (pest, _, _) = fit_Sinewave_to_Firingrate(firingRate[jFit], freq[i], glr_dt)
        Gain[i]  = pest[2] #/(amplitude*2π*freq[i])  # gain spikes/sec per deg/sec (?? check)
        Phase[i] = pest[3]   
        DC[i]    = pest[1]    

        # splot!(ax,spiketime)
        # lines!(sampleTime, firingRate)

        # F

        #@infiltrate

        # for j = 1:Nfit

        #     # indices of sampleTime containing jth response cycle
        #     t0 = (Nburn + j - 1)*period  # jth response cycle starts here
        #     jthCycle = findall(t0 .<= sampleTime .<= (t0+period))

        #     #@infiltrate

        #     # fit sinewave to jth response cycle, pest = (offset, amplitude, phase)
        #     (pest, _, _) = fit_Sinewave_to_Firingrate(firingRate[jthCycle], freq[i], glr_dt)
 
        # Gain[i,j] = pest[2] #/(amplitude*2π*freq[i])  # gain spikes/sec per deg/sec (?? check)
        # Phase[i, j] = pest[3]   # radians to degrees
        # #@infiltrate
        # end
        #println(i, ", ", freq[i], ", ", i/length(freq))
        print(".")  # indicator to show that something is happening ...
    end
    println("")
    (freq, Gain, Phase, DC)

end

# Bode gain and phase plots 
# freq = 1 x Nfreqs 
# Gain, Phase = Nfreqs x Nreps array
function drawBodePlots(freq, Gain, Phase)

    # data dimensions not cross-checked, will crash with error anyway
    Nfreqs = length(freq)
    Nreps  = 1 #size(Gain)[2] 

    # # average gain and phase
    # avg_Gain = mean(Gain, dims=2)[:]
    # avg_Phase = mean(Phase, dims=2)[:]

    Fig = Figure(size = (800,600))
    #ax1 = Axis(Fig[1,1:2])
    ax2 = Axis(Fig[1,1], xscale = log10, yscale = log10)
    ax3 = Axis(Fig[2,1], xscale = log10) #, yticks = [-40., -20.0, 0.0, 20., 40.0])
    #ylims!(ax3, [-45.0, 45.0])
    
    lines!(ax2, freq, Gain)
    lines!(ax3, freq, -Phase*180.0/π)
    
    #ylims!(ax2, [0.1, 10.])
    
    display(Fig)
    
   # save("Exwald_Sinewave_Bode.png", Fig)
    
    Fig
end

# NB f in Hz
function sinewave(p, f, t)

    p[1] .+ p[2]*sin.(2.0*π*f*t .- p[3])

end

# derivative of function of t by central difference
# usage e.g.:
#    t = t=0.0:dt:5.0
#    f = t-> sinewave((0.0, 1.0, 0.0), 1.0, t)  # f is a function of t (sin(2πt) in this e.g.)
#   df = t->diffcd(f, t)   # df is a function of t, returns the derivative of f at t (2πcos(2πt) here)
#   These functions can be broadcast over vectors or ranges of t: x = f.(t), dx = df.(t)
function diffcd(f::Function, t::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    return (f(t+dt) - f(t-dt))/(2*dt)

end

# central difference differentiator for vector of values
function diffcd(v::Vector{Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    N = length(v)
    dv = zeros(Float64, N)
    dv[1] = (v[2]-v[1])/dt  # one-sided right derivative
    for i in 2:(N-1)
        dv[i] = (v[i+1] - v[i-1])/(2*dt)  # central
    end
    dv[N] = (v[N] - v[N-1])/dt  # one-sided left

    return dv
end

# fractional differintegrator by convolving vector f with power law kernel.
# Because dq is history-dependent (i.e. current state vector includes all previous inputs)
# it is prohibitively slow to compute dq() by updating at each time step. 
# This is not fixable in a general way because e.g. for q=-1 (integral) the effect of 
# past inputs does not decay over time. 
function dq(f::Vector{Float64}, q, dt::Float64=DEFAULT_SIMULATION_DT)

    N = length(f);
    T = 1:N;
    K = cumprod(pushfirst!((T.-1.0.-q)./T, 1.0));
    dt^-q*conv(K,f)[1:N];

end

# returns closure to compute fractional differintegrator d^q u(t) / dt^q
# q > 0.0 for fractional differentiator, q < 0.0 for integrator 
# Numerical approximation for input signal u_k with sample interval dt
# using Oustaloop method in band (f0, f1) Hz (approximate the operator using linear transfer function) 
# Default bandwidth is 1/minute to 1/10 of Nyquist Freq
function make_fractional_derivative(q::Float64, f0::Float64=1.0/60.0, f1::Float64 = 0.1*π/DEFAULT_SIMULATION_DT,
         dt::Float64=DEFAULT_SIMULATION_DT)

    # Approximation of order 2N+1 (so N=2 is 5th order)
    N = 4
    
    # Oustaloup transfer function parameters
    K, poles, xeros = oustaloup_zeros_poles(q, N, 2π*f0, 2π*f1)
    
    # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
    residues, _ = oustaloup_residues(K, poles, xeros)
    p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
    M = length(poles)  # ... = 2N+1

    # state: x = [v1, ..., vM]
    x = zeros(Float64,M)
    dx = zeros(Float64,M)

    # fractional update at t given u(t)
    function dq(u::Float64)

        # Approximate D^q u 
        if (q==0.0) 
            D = u
        else
            D = K * u
            for i in 1:M
                D += residues[i] * x[i]
            end
        end
        
        # internal state update
        for i in 1:M
            x[i] += (-p_i[i] * x[i] + u)*dt
        end

        return D
    end

    # return closure
    return dq 
end

# add a tuple or a vector to a tuple 
function tpadd(T::Tuple, t::Union{Tuple, Vector})

    @assert length(T)==length(t) "addends must be the same length"

    return tuple(vec(collect(T) + collect(t))...)

end



# return trigger threshold for mean interval tau between threshold-crossing events 
# for specified noise Normal(μ,s)
function Poisson_Trigger_Threshold(τ::Float64, μ::Float64=0.0, s::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    quantile(Normal(μ, s), 1.0 - dt / τ)  # returns threshold 

end

# return mean noise level μ to get mean interval tau between threshold-crossing events 
# with noise input Normal(μ,s) for specified s and Threshold
function mean_noise_for_Poisson_given_threshold(τ::Float64, s::Float64, Threshold::Float64, 
                                            dt::Float64=DEFAULT_SIMULATION_DT)

    z = quantile(Normal(0.0, 1), 1.0 - dt / τ)  # z for required tail probability
    μ = Threshold - s*z                           # put the mean that far below the threshold

end

# find mean interval length of threshold trigger with input noise Normal(m,s)
function PoissonTau_from_ThresholdTrigger(m::Float64, s::Float64, threshold::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    dt / (1.0 - cdf(Normal(m, s), threshold))

end

        
# return CV of Exwald model neuron
function CV_fromExwaldModel(mu::Float64, lambda::Float64, tau::Float64)


    waldVariance = mu^3/lambda 
    expVariance =tau^2

    exwaldVariance = expVariance + waldVariance
    exwaldMean = mu + tau

    return sqrt(exwaldVariance)/exwaldMean

end

# mean, sd, cv and cv* of numerical pdf
function pdf_stats(grid::Vector{Float64}, pdf_vals::Vector{Float64})
    # Trapezoidal integration
    dg   = diff(grid)
    # mean = ∫ t·f(t) dt
    m    = sum((grid[1:end-1] .* pdf_vals[1:end-1] .+ 
                grid[2:end]   .* pdf_vals[2:end]) ./ 2.0 .* dg)
    # E[t²] = ∫ t²·f(t) dt
    et2  = sum((grid[1:end-1].^2 .* pdf_vals[1:end-1] .+ 
                grid[2:end].^2   .* pdf_vals[2:end])   ./ 2.0 .* dg)
    var  = et2 - m^2
    sd = sqrt(var)
    cv   = sd / m

    cvstar = cvStar(m, cv)

    return m, sd, cv, cvstar
end

# CV* from mean and CV
# Goldberg, Smith and Fernandez (1984)
function cvStar(m::Float64, cv::Float64)

    # Goldberg, Smith and Fernandez (1984) TABLE 1
    # They used milliseconds but we use seconds
    tbar_tab = vec([5.0   5.5  6.0   6.5   7.0   7.5   12.5  17.5  22.5  27.5 32.5  37.5 42.5  47.5  52.5])
    a_tab =    vec([0.36  0.4  0.46  0.53  0.55  0.56  0.84  1.15  1.49  1.66  1.68  1.8  1.82  1.88  1.93])
    b_tab =    vec([0.63  0.66  0.73  0.79  0.8  0.81  0.97  1.02  1.04  1.01  0.96  0.93  0.91  0.9  0.89])

    # seconds to ms
    tbar = m*1000.0 

       # interpolators (using BasicInterpolators.jl)
    if (tbar<tbar_tab[1] || tbar>tbar_tab[end])  # model is out of bounds, use linear extrapolation
        a_interpolate = LinearInterpolator(tbar_tab, a_tab, NoBoundaries())
        b_interpolate = LinearInterpolator(tbar_tab, b_tab, NoBoundaries())
    else
        a_interpolate = CubicSplineInterpolator(tbar_tab, a_tab)
        b_interpolate = CubicSplineInterpolator(tbar_tab, b_tab)
    end

    # interpolated coefficients 
    a = a_interpolate(tbar)
    b = b_interpolate(tbar)

    return (cv/a)^(1.0/b)

end


# CV* of Exwald model test_fit_Exwald_neuron_stationary
function CVStar_fromExwaldModel(mu::Float64, lambda::Float64, tau::Float64)

    # Goldberg, Smith and Fernandez 1984 TABLE 1
    # nb This table uses milliseconds but (our) model convention is seconds
    tbar_tab = vec([5.0   5.5  6.0   6.5   7.0   7.5   12.5  17.5  22.5  27.5 32.5  37.5 42.5  47.5  52.5])
    a_tab =    vec([0.36  0.4  0.46  0.53  0.55  0.56  0.84  1.15  1.49  1.66  1.68  1.8  1.82  1.88  1.93])
    b_tab =    vec([0.63  0.66  0.73  0.79  0.8  0.81  0.97  1.02  1.04  1.01  0.96  0.93  0.91  0.9  0.89])

    s2ms = 1000.0

    # mean interval length for this neuron
    tbar = s2ms*(mu + tau)

    # CV for this neuron
    cv =  CV_fromExwaldModel(mu, lambda, tau)

    # interpolators (using BasicInterpolators.jl)
    if (tbar<tbar_tab[1] || tbar>tbar_tab[end])  # model is out of bounds, use linear extrapolation
        a_interpolate = LinearInterpolator(tbar_tab, a_tab, NoBoundaries())
        b_interpolate = LinearInterpolator(tbar_tab, b_tab, NoBoundaries())
    else
        a_interpolate = CubicSplineInterpolator(tbar_tab, a_tab)
        b_interpolate = CubicSplineInterpolator(tbar_tab, b_tab)
    end

    # interpolated coefficients 
    a = a_interpolate(tbar)
    b = b_interpolate(tbar)

    return (cv/a)^(1.0/b)

end


# CV* given ISI
function CVStar(ISI::Vector{Float64})

    # Goldberg, Smith and Fernandez 1984 TABLE 1
    # nb This table uses milliseconds but (our) model convention is seconds
    tbar_tab = vec([5.0   5.5  6.0   6.5   7.0   7.5   12.5  17.5  22.5  27.5 32.5  37.5 42.5  47.5  52.5])
    a_tab =    vec([0.36  0.4  0.46  0.53  0.55  0.56  0.84  1.15  1.49  1.66  1.68  1.8  1.82  1.88  1.93])
    b_tab =    vec([0.63  0.66  0.73  0.79  0.8  0.81  0.97  1.02  1.04  1.01  0.96  0.93  0.91  0.9  0.89])

    # mean interval length in ms 
    tbar = mean(ISI)*1000.0

    # CV for this neuron
    CV =  std(ISI)/mean(ISI)

    # interpolators (using BasicInterpolators.jl)
    if (tbar<tbar_tab[1] || tbar>tbar_tab[end])  # model is out of bounds, use linear extrapolation
        a_interpolate = LinearInterpolator(tbar_tab, a_tab, NoBoundaries())
        b_interpolate = LinearInterpolator(tbar_tab, b_tab, NoBoundaries())
    else
        a_interpolate = CubicSplineInterpolator(tbar_tab, a_tab)
        b_interpolate = CubicSplineInterpolator(tbar_tab, b_tab)
    end

    # interpolated coefficients 
    a = a_interpolate(tbar)
    b = b_interpolate(tbar)

    return (CV/a)^(1.0/b)

end



function Exwald_fromCV(CV::Float64)

    # Slope and intercept of PC1 in log τ-λ axes
    # nb time in milliseconds
    log_tau_0 = 0.0                     # log tau = 0, ie "y-axis" on log-log plot
    log_lambda_0 = 3.2031               # intercept, lambda @ log tau = 0 
    d_loglambda_logtau = -0.60277   # slope in log-log axes
    mu_0 = 12.191

    # pick two initial points at extremes of the distribution on PC1
    # and evaluate CV at these points
    log_tau_a = -2.0
    log_lambda_a = log_lambda_0 + (log_tau_a - log_tau_0)*d_loglambda_logtau
    CV_a = CV_fromExwaldModel(mu_0, 10.0^log_lambda_a, 10.0^log_tau_a)   

    log_tau_b = 2.0
    log_lambda_b = log_lambda_0 + (log_tau_b - log_tau_0)*d_loglambda_logtau
    CV_b = CV_fromExwaldModel(mu_0, 10.0^log_lambda_b, 10.0^log_tau_b)

    # bisection search to find point on PC1 with the specified CV
    # nb by construction CV_b > CV_a
    while abs(CV_a-CV_b) > .001


        log_tau_c = (log_tau_a + log_tau_b)/2.0
        log_lambda_c = (log_lambda_b + log_lambda_a)/2.0

        CV_c = CV_fromExwaldModel(mu_0, 10.0^log_lambda_c, 10.0^log_tau_c)

        if CV_c > CV     # CV must lie between point a and c, replace b with c
            log_tau_b = log_tau_c 
            log_lambda_b = log_lambda_c 
            CV_b = CV_c
        else             # CV must lie between point c and b, replace a with c
            log_tau_a = log_tau_c 
            log_lambda_a = log_lambda_c 
            CV_a = CV_c
        end

    end

    # convert to seconds for return
    ms2s = 0.001
    log_lambda = (log_lambda_a + log_lambda_b)/2.0
    log_tau    = (log_tau_a + log_tau_b)/2.0
    return ms2s*mu_0, ms2s*10.0^log_lambda, ms2s*10.0^log_tau

end

function Exwald_fromCVStar(CVStar::Float64, tbar::Float64)
#TBD

end

# First passage time model parameters (v,s,barrier) from Wald parameters (mu, lambda)
# One FPT parameter must be specified by name and value. 
# Name choices are "noiseMean", "noiseSD" and "barrier". 
# Default name is "barrier", default value is 1.0 
function FirstPassageTime_parameters_from_Wald(mu::Float64, lambda::Float64,
    specifiedName::String="barrier", specifiedValue::Float64=1.0)

    @match specifiedName begin   # macro in MLStyle.jl
        "noiseMean" => return (specifiedValue, specifiedValue * mu / sqrt(lambda), specifiedValue * mu)
        "noiseSD" => return (specifiedValue * sqrt(lambda) / mu, specifiedValue, specifiedValue * sqrt(lambda))
        "barrier" => return (specifiedValue / mu, specifiedValue / sqrt(lambda), specifiedValue)
    end
end

# FPT model (v,s,barrier) defines unique Wald distribution (mu, lambda)
function Wald_parameters_from_FirstpassageTimeModel(v::Float64, s::Float64, barrier::Float64)

    (barrier / v, (barrier / s)^2)

end

# --- Example Usage ---
# Generate some dummy data
# N = 50
# A = rand(N, 3)
# B = A .+ 0.2 .* randn(N, 3) # B is a noisy version of A

# # Call the function
# fig = plot_linked_points(A, B)
# display(fig)

function findvrange(mu_0, taus)

    vmin = Inf
    vmax = -Inf

    for i in 1:length(mu_0)
        for k in 1:length(taus)
            v = 1.0/(1.0 - exp(-mu_0[i]/taus[k]))
            if v>vmax
                vmax = v
            end
            if v<vmin
                vmin = v
            end
        end
    end

    return vmin, vmax

end

# v0 parameter of SLIF model required to get mean interval length mu 
# given time constant tau (with threshold 1)
# i.e. input current required for deterministic leaky integrator 
# with time constant tau to reach threshold
function SLIF_v0(mu::Float64, tau::Float64)

    return 1.0/(1-exp(-mu/tau))

end

# ISI distribution
# returns bin_count, relative_frequency, bin_edges and bin centres given ISIs 
# can specify maxT and ( nbins or binwidth ) 
# NB nbins over-rides binwidth if both are specified
#    (i.e. that's an error but I'll let you off with a warning)
function ISI_distribution(ISI::Vector{Float64}; 
                            maxT::Float64 = -1.0, nbins::Int64= -1, bw::Float64 = -1.0)

    # if extent of histogram is not specified, choose maxT large enough to cover the data
    if maxT<0.0
        maxT = 1.25*maximum(ISI) 
    end

    # figure out binwidth and number of bins 
    if nbins<0                  
        if bw < 0.0             
            nbins = 32              # nbins not specified, bw not specified -> set default nbins
            bw = maxT/nbins         #                                       -> compute bw
        else 
            nbins = ceil(maxT/bw)   # nbins not specified, bw specified -> compute nbins 
            maxT = bw*nbins         # adjust maxT to encompass a whole number of bins
        end
    else
        if bw>0.0
            @warn "nbins over-rides binwidth when both are specified"
        end
        bw = maxT/nbins             # nbins specified -> compute bw even if it was specified
    end

    bin_edges = collect(0.0:bw:maxT)

    H = fit(Histogram, ISI, bin_edges) # uses Statsbase.jl

    # relative frequencies in bins 
    freqs = H.weights./(sum(H.weights)*bw)

    bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)

    # return counts, frequencies, bin edges and bin centres
    return H.weights, freqs, bin_edges, bin_centres

end

# simulate the effect of finite sample interval on observed spike times
# Our data were obtained at 300us resolution
function quantize_intervals(ISI::Vector{Float64}, samplePeriod::Float64=300.0e-6)

    st = cumsum(ISI)                            # spike times from intervals
    qst = ceil.(st/samplePeriod)*samplePeriod    # quantized spike times

    return diff([0.0; qst])                     # return quantized intervals
end

# n points logarithmically spaced from f0 to f1
logspace(f0, f1, n) = exp10.(range(log10(f0), log10(f1), length=n))

# draw Bode plots, Gain and Phase of t->response(sin(2πft))
# over bandwidth specified in Hz. 
# Gain & phase averaged over Ncycles after burn-in time
# function BodePlot(response::Function, f0::Float64 = .01, f1::Float64 = 25.0,
function BodePlot(response::Function, title::String = "Chinchilla Cupula Displacement", A::Float64 = 1.0, 
                  f0::Float64 = 0.01, f1::Float64 = 25.0, 
                  nf::Int64 = 16, Ncycles::Int64 = 8, 
                  burntime::Float64 = 1/f0;
                  dt::Float64 = DEFAULT_SIMULATION_DT)

    freqs  = exp10.(range(log10(f0), log10(f1), length=nf))
    gains  = zeros(nf)
    phases = zeros(nf)

    for (i, f) in enumerate(freqs)

        period   = 1.0 / f
        meastime = Ncycles * period
        T        = burntime + meastime
        n_burn   = round(Int, burntime / dt)
        n_meas   = round(Int, meastime / dt)
        n_total  = n_burn + n_meas

        # Run simulation
        t_vec = (1:n_total) .* dt
        u     = t -> A * sin(2π * f * t)
        out   = zeros(n_total)
        for k in 1:n_total
            y = response(u, t_vec[k])
            out[k] = y isa Tuple ? y[1] : y   # because some response functions return state not output
        end

        # Discard burn-in, keep measurement window
        y = out[n_burn+1:end]
        t = t_vec[n_burn+1:end]

        # Fit sine at stimulus frequency by linear regression
        # y ≈ C1*sin(2πft) + C2*cos(2πft)
        # → gain  = sqrt(C1² + C2²) / A
        # → phase = atan(-C2, C1)   [relative to input sin]
        S  = sin.(2π .* f .* t)
        C  = cos.(2π .* f .* t)
        # Least squares: [S C] \ y
        M       = hcat(S, C)
        coeffs  = M \ y
        C1, C2  = coeffs[1], coeffs[2]

        gains[i]  = sqrt(C1^2 + C2^2) / A
        phases[i] = atan(C2, C1)      # phase lag in radians

        @printf("f=%6.3f Hz   gain=%8.4f   phase=%7.3f rad (%6.1f deg)\n",
                f, gains[i], phases[i], rad2deg(phases[i]))
    end

    # normalize gains (maximum gain = 1.0)
    gains = gains/maximum(gains)

    # ── Bode plot ────────────────────────────────────────────────────────────
    gain_dB = 20 .* log10.(gains)

    fig = Figure(size=(800, 600))

    ax1 = Axis(fig[1,1],
               xscale      = log10,
               ylabel      = "Normalized Gain (dB)",
               title       = title,
               xticksvisible = false,
               xticklabelsvisible = false)

    ax2 = Axis(fig[2,1],
               xscale  = log10,
               xlabel  = "Frequency (Hz)",
               ylabel  = "Phase (deg)")

    lines!(ax1, freqs, gain_dB;  color=:steelblue, linewidth=2)
    scatter!(ax1, freqs, gain_dB; color=:steelblue, markersize=4)

    lines!(ax2, freqs, rad2deg.(phases);  color=:crimson, linewidth=2)
    scatter!(ax2, freqs, rad2deg.(phases); color=:crimson, markersize=4)

    # Zero dB and zero phase reference lines
    hlines!(ax1, [0.0]; linestyle=:dash, color=:gray, linewidth=1)
    hlines!(ax2, [0.0]; linestyle=:dash, color=:gray, linewidth=1)

    rowsize!(fig.layout, 1, Relative(0.5))
    display(fig)

    return freqs, gains, phases, fig
end


# draw Bode plots, Gain and Phase of t->response(sin(2πft)) for spiking neuron model
# over bandwidth specified in Hz. 
# Gain & phase averaged over Ncycles after burn-in time
# function BodePlot(response::Function, f0::Float64 = .01, f1::Float64 = 25.0,
function spikingneuron_BodePlot(neuron::Function, title::String = "Chinchilla Cupula Displacement", A::Float64 = 1.0, 
                  f0::Float64 = 0.01, f1::Float64 = 25.0, 
                  nf::Int64 = 16, Ncycles::Int64 = 8, 
                  burntime::Float64 = 1/f0;
                  dt::Float64 = DEFAULT_SIMULATION_DT)

    freqs  = exp10.(range(log10(f0), log10(f1), length=nf))
    gains  = zeros(nf)
    phases = zeros(nf)

    for (i, f) in enumerate(freqs)

        period   = 1.0 / f
        meastime = Ncycles * period
        T        = burntime + meastime
        n_burn   = round(Int, burntime / dt)
        n_meas   = round(Int, meastime / dt)
        n_total  = n_burn + n_meas

        # Simulate neuron and estimate rate at stimulus bandwidth
        t_vec = (1:n_total) .* dt
        u     = t -> A * sin(2π * f * t)
        out   = GLR(spiketimes(neuron, u, T), f, T)

        # Discard burn-in, keep measurement window
        y = out[n_burn+1:end]
        t = t_vec[n_burn+1:end]

        # Fit sine at stimulus frequency by linear regression
        # y ≈ C1*sin(2πft) + C2*cos(2πft)
        # → gain  = sqrt(C1² + C2²) / A
        # → phase = atan(-C2, C1)   [relative to input sin]
        S  = sin.(2π .* f .* t)
        C  = cos.(2π .* f .* t)
        # Least squares: [S C] \ y
        M       = hcat(S, C)
        coeffs  = M \ y
        C1, C2  = coeffs[1], coeffs[2]

        gains[i]  = sqrt(C1^2 + C2^2) / A
        phases[i] = atan(C2, C1)      # phase lag in radians

        @printf("f=%6.3f Hz   gain=%8.4f   phase=%7.3f rad (%6.1f deg)\n",
                f, gains[i], phases[i], rad2deg(phases[i]))
    end

    # normalize gains (maximum gain = 1.0)
    gains = gains/maximum(gains)

    # ── Bode plot ────────────────────────────────────────────────────────────
    gain_dB = 20 .* log10.(gains)

    fig = Figure(size=(800, 600))

    ax1 = Axis(fig[1,1],
               xscale      = log10,
               ylabel      = "Normalized Gain (dB)",
               title       = title,
               xticksvisible = false,
               xticklabelsvisible = false)

    ax2 = Axis(fig[2,1],
               xscale  = log10,
               xlabel  = "Frequency (Hz)",
               ylabel  = "Phase (deg)")

    lines!(ax1, freqs, gain_dB;  color=:steelblue, linewidth=2)
    scatter!(ax1, freqs, gain_dB; color=:steelblue, markersize=4)

    lines!(ax2, freqs, rad2deg.(phases);  color=:crimson, linewidth=2)
    scatter!(ax2, freqs, rad2deg.(phases); color=:crimson, markersize=4)

    # Zero dB and zero phase reference lines
    hlines!(ax1, [0.0]; linestyle=:dash, color=:gray, linewidth=1)
    hlines!(ax2, [0.0]; linestyle=:dash, color=:gray, linewidth=1)

    rowsize!(fig.layout, 1, Relative(0.5))
    display(fig)

    return freqs, gains, phases, fig
end

function reposition_legend!(fig::Figure, position)
    ax = first(x for x in fig.content if x isa Axis)
    for element in copy(fig.content)
        element isa Legend && delete!(element)
    end
    axislegend(ax; position=position)
    display(F)
end

function phase_histogram_inset!(fig, ax1, spt::Vector{Float64}, f::Float64, T_spont::Float64;
                                 inset_x::Float64    = 0.05,    # left edge of inset
                                 inset_y::Float64    = 0.6,    # bottom edge of inset
                                 inset_w::Float64    = 0.12,   # width
                                 inset_h::Float64    = 0.36,   # height
                                 n_sectors::Int      = 24,
                                 title::String       = "",
                                 markercolor         = :steelblue)

    stim_spikes = filter(t -> t >= T_spont, spt)
    n_spikes    = length(stim_spikes)
    n_spikes == 0 && (@warn "No spikes during stimulus"; return nothing)

    phases_rad = mod.(2π .* f .* (stim_spikes .- T_spont) .- π/2, 2π)

    sector_width_rad = 2π / n_sectors
    counts           = zeros(n_sectors)
    for φ in phases_rad
        i = floor(Int, φ / sector_width_rad) + 1
        counts[clamp(i, 1, n_sectors)] += 1
    end

    # Log scaling — uniform = 0.5
    expected = n_spikes / n_sectors
    radii    = 0.5 .* (1.0 .+ log.(max.(counts, 0.5) ./ expected) ./ log(n_sectors))
    radii    = clamp.(radii, 0.0, 1.0)

    R          = sqrt(mean(sin.(phases_rad))^2 + mean(cos.(phases_rad))^2)
    mean_phase = atan(mean(sin.(phases_rad)), mean(cos.(phases_rad)))

    # ── Convert fractional ax1 coordinates to figure bbox ───────────────────
    # Get ax1 pixel bounds
    ax1_scene  = ax1.scene
    px         = ax1_scene.viewport[]
    ax1_left   = px.origin[1]
    ax1_bottom = px.origin[2]
    ax1_width  = px.widths[1]
    ax1_height = px.widths[2]

    # Convert 0-1 fractions to pixel coordinates
    left   = ax1_left   + inset_x * ax1_width
    bottom = ax1_bottom + inset_y * ax1_height
    width  = inset_w * ax1_width
    height = inset_h * ax1_height

    # Create inset axis with absolute pixel bbox
    ax_inset = Axis(fig;
                    bbox    = Makie.BBox(left, left+width, bottom, bottom+height),
                    aspect  = DataAspect(),
                    title   = title,
                    titlesize = 10,
                    xticksvisible = false,
                    yticksvisible = false,
                    xgridvisible = false,
                    ygridvisible = false,
                    xticklabelsvisible = false,
                    yticklabelsvisible = false,
                    leftspinevisible   = false,
                    rightspinevisible  = false,
                    topspinevisible    = false,
                    bottomspinevisible = false)

    limits!(ax_inset, -1.4, 1.4, -1.4, 1.4)

    # ── Draw polar histogram ─────────────────────────────────────────────────
    θ_circle = range(0, 2π, length=200)

    # Reference circles
    for r in [0.25, 0.5, 0.75, 1.0]
        if r == 0.5
            lines!(ax_inset, r .* sin.(θ_circle), r .* cos.(θ_circle);
                   color=:red, linewidth=1.5, linestyle=:dot)
        else
            lines!(ax_inset, r .* sin.(θ_circle), r .* cos.(θ_circle);
                   color=(:gray, 0.3), linewidth=0.5)
        end
    end

    # Radial lines and labels at 90° intervals only (keep it clean at small size)
    for θ_deg in 0:90:270
        θ = deg2rad(θ_deg)
        lines!(ax_inset, [0.0, sin(θ)], [0.0, cos(θ)];
               color=(:gray, 0.3), linewidth=0.5)
        label = @sprintf "%d°" θ_deg
        text!(ax_inset, 1.3*sin(θ), 1.3*cos(θ);
              text=label, align=(:center, :center), fontsize=8)
    end

    # Sectors
    for i in 1:n_sectors
        r = radii[i]
        r == 0.0 && continue
        θ_lo    = (i-1) * sector_width_rad
        θ_hi    =  i    * sector_width_rad
        θ_range = range(θ_lo, θ_hi, length=20)
        xs = vcat(0.0, r .* sin.(θ_range), 0.0)
        ys = vcat(0.0, r .* cos.(θ_range), 0.0)
        poly!(ax_inset, Point2f.(xs, ys);
              color=(markercolor, 0.7), strokecolor=:white, strokewidth=0.5)
    end

    # Mean phase arrow
    lines!(ax_inset, [0.0, R*sin(mean_phase)], [0.0, R*cos(mean_phase)];
           color=:crimson, linewidth=2.0)
    scatter!(ax_inset, [R*sin(mean_phase)], [R*cos(mean_phase)];
             color=:crimson, markersize=8)

    # # R and n annotation
    # text!(ax_inset, 0.0, -1.35;
    #       text=@sprintf("R=%.2f n=%d", R, n_spikes),
    #       align=(:center, :center), fontsize=8)

    display(fig)
    return ax_inset
end

# Vector{Vector{Float64}} method
function phase_histogram_inset!(fig, ax1, spt::Vector{Vector{Float64}}, f::Float64,
                                 T_spont::Float64; kwargs...)
    phase_histogram_inset!(fig, ax1, vcat(spt...), f, T_spont; kwargs...)
end

# phase histogram in axes
function phase_histogram!(ax, spt::Vector{Float64}, f::Float64, T_spont::Float64;
                                 n_sectors::Int      = 24,
                                 title::String       = "",
                                 markercolor         = :steelblue)

    stim_spikes = filter(t -> t >= T_spont, spt)
    n_spikes    = length(stim_spikes)
    n_spikes == 0 && (@warn "No spikes during stimulus"; return nothing)

    phases_rad = mod.(2π .* f .* (stim_spikes .- T_spont) .- π/2, 2π)

    sector_width_rad = 2π / n_sectors
    counts           = zeros(n_sectors)
    for φ in phases_rad
        i = floor(Int, φ / sector_width_rad) + 1
        counts[clamp(i, 1, n_sectors)] += 1
    end

    # Log scaling — uniform = 0.5
    expected = n_spikes / n_sectors
    radii    = 0.5 .* (1.0 .+ log.(max.(counts, 0.5) ./ expected) ./ log(n_sectors))
    radii    = clamp.(radii, 0.0, 1.0)

    R          = sqrt(mean(sin.(phases_rad))^2 + mean(cos.(phases_rad))^2)
    mean_phase = atan(mean(sin.(phases_rad)), mean(cos.(phases_rad)))

    # ── Draw polar histogram ─────────────────────────────────────────────────
    θ_circle = range(0, 2π, length=200)

    # Reference circles
    for r in [0.25, 0.75]
        lines!(ax, r .* sin.(θ_circle), r .* cos.(θ_circle);
                    color=(:gray, 0.3), linewidth=0.5)                  
    end

    # dotted red line = expected value for uniform distribution
    lines!(ax, 0.5 .* sin.(θ_circle), 0.5 .* cos.(θ_circle);
                    color=:red, linewidth=1.5, linestyle=:dot)
    
    # solid line = 100% lock
    lines!(ax, 1.0 .* sin.(θ_circle), 1.0 .* cos.(θ_circle);
                    color=:black, linewidth=1)

    # Radial lines and labels at 90° intervals only (keep it clean at small size)
    for θ_deg in [0, 90, -90]
        θ = deg2rad(θ_deg)
        lines!(ax, [0.0, sin(θ)], [0.0, cos(θ)];
               color=(:gray, 0.3), linewidth=0.5)
        label = @sprintf "%d°" θ_deg
        text!(ax, 1.3*sin(θ), 1.3*cos(θ);
              text=label, align=(:center, :center), fontsize=14)
    end

    # Sectors
    for i in 1:n_sectors
        r = radii[i]
        r == 0.0 && continue
        θ_lo    = (i-1) * sector_width_rad
        θ_hi    =  i    * sector_width_rad
        θ_range = range(θ_lo, θ_hi, length=20)
        xs = vcat(0.0, r .* sin.(θ_range), 0.0)
        ys = vcat(0.0, r .* cos.(θ_range), 0.0)
        poly!(ax, Point2f.(xs, ys);
              color=(markercolor, 0.7), strokecolor=:white, strokewidth=0.5)
    end

    # Mean phase arrow
    lines!(ax, [0.0, R*sin(mean_phase)], [0.0, R*cos(mean_phase)];
           color=:crimson, linewidth=2.0)
    scatter!(ax, [R*sin(mean_phase)], [R*cos(mean_phase)];
             color=:crimson, markersize=8)

    text!(ax, 0.48, 0.9, 
      text = " lead ←",
      space = :relative,
      align = (:right, :top),
      fontsize = 14,
      color = :black)

    text!(ax, 0.52, 0.9, 
      text = "→ lag",
      space = :relative,
      align = (:left, :top),
      fontsize = 14,
      color = :black)

    # # R and n annotation
    # text!(ax_inset, 0.0, -1.35;
    #       text=@sprintf("R=%.2f n=%d", R, n_spikes),
    #       align=(:center, :center), fontsize=8)

end

# Vector{Vector{Float64}} method
function phase_histogram!(ax, spt::Vector{Vector{Float64}}, f::Float64, T_spont::Float64)
    phase_histogram!(ax, vcat(spt...), f, T_spont)
end