# Neuralian Toolbox Utilities
# misc tools including pdfs
# MGP 2024-25


# plot spiketime vector as spikes
function splot(spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0)

    linesegments(vec([( Point2f(t, 0.0), Point2f(t, height)) for t in spiketime]),
        linewidth = lw, color = :blue)

end


function splot!(ax::Axis, spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0, color = :blue)

    spikes = linesegments!(ax, vec([( Point2f(t, 0.0), Point2f(t, height)) for t in spiketime]),
        linewidth = lw, color = color)
    baseline = lines!(ax, [0.0, spiketime[end]], [0.0, 0.0], color = color)
    (spikes, baseline)
end

# Uniform Gaussian rate filter
# Firing rate estimation by sampling Gaussians centred at spike times
#   spiketimes, sd of Gaussian filter, dt = sampling interval, T = end time
# MGP July 2024
# spiketime in seconds
# sd of Gaussian kernal for each spike (ifP length(sd)==1 then same kernel for all spikes)
# dt = sample interval in seconds
# pad = pad the start and end of the spike train with mirror image of spikes in first and last pad seconds 
#       (kluge to prevent edge effects. the padded ends are removed before return) (default no pads).
# last sample time (default rounded up to nearest second)
function GLR(spiketime::Vector{Float64}, sd::Vector{Float64}, dt::Float64, pad::Float64=0.0, T::Float64=maximum(spiketime))

    if length(sd) == 1
        sd = sd[] * ones(length(spiketime))
    end

    if length(sd) != length(spiketime)
        print("length mismatch: length of sd must equal number of spikes")
        return
    end


    # print("length(spiketime) = "), println(length(spiketime))
    # print("pad= "), println(pad)

    if pad > 0.0
        ifront = findall(spiketime[:] .< pad)[end:-1:1]  # indices of front pad spikes in reverse order (mirror) 
        iback = findall(spiketime[:] .> (T - pad))[end:-1:1]
        #infiltrate
        spiketime = pad .+ vcat(2.0 * spiketime[1] .- spiketime[ifront[1:end-1]], spiketime, 2.0 * spiketime[iback[1]] .- spiketime[iback[2:end]])
        sd = vcat(sd[ifront[2:end]], sd, sd[iback[2:end]])
    end
    # print("length(spiketime) = "), println(length(spiketime))

    # vector to hold result
    padN = Int(ceil(pad / dt))    # pad lengths
    sigN = Int(ceil(T / dt))      # signal length (number of sample points)
    N = sigN + 2 * padN           # padded signal length
    t = (1:N) * dt                  # time vector  (for return)
    r = zeros(N)                  # rate vector


    for i in 1:length(spiketime)
        k = Int(round(spiketime[i] / dt)) # ith spike occurs at this sample point
        n = Int(round(4.0 * sd[i] / dt))  # number of sample points within 4 sd each side of spike
        kernel = pdf(Normal(spiketime[i], sd[i]), k * dt .+ (-n:n) * dt)
        kernel = kernel/sum(kernel*dt)  # normalize (each spike contributes power 1)
        for j in -n:n  # go 4 sd each side
            if (k + j) >= 1 && (k + j) <= N    # in bounds 
                r[k+j] += kernel[n+j+1]
            end
        end
    end

    if padN>0
        t0 = t[padN]
    else
        t0 = 0.0
    end

    return (t[padN.+(1:sigN)].-t0, r[padN.+(1:sigN)])


end

# utility alias for GLR (because usually length(sd)==1)
function GLR(spiketime::Vector{Float64}, sd::Float64, dt::Float64, pad::Float64=0.0, T::Float64=maximum(spiketime))
    GLR(spiketime, [sd], dt, pad, T)
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

# Kullback-Liebler divergence from ISI data to model
# see Paulin, Pullar and Hoffman (2024) Sec. 2.2.3
function sKLD(interval::Vector{Float64}, model::Function)

    N = length(interval)
    return -sum(log2.([model(interval[i]) for i in 1:N]))/N + log2(N)
end


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
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Float64)

    @assert lambda > 0.0    "lambda must be > 0.0"
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

# Exwald pdf parameterized by cv
function Exwaldpdf_cv(mu::Float64, cv::Float64, tau::Float64, t::Float64)

   Exwaldpdf(mu, mu/cv^2, tau, t)

end

# Exwald pdf parameterized by cv
function Exwaldpdf_cv(mu::Float64, cv::Float64, tau::Float64, t::Vector{Float64}, P::Bool=false)

   Exwaldpdf(mu, mu/cv^2, tau, t, P)

end



# Exwald pdf at vector of times
# renormalized if renorm==true
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, renorm::Bool=false)

    p = [Exwaldpdf(mu, lambda, tau, s) for s in t]
    if renorm
        p[findall(p.<0.0)] .=0.0  
        p /= (sum(p)*(t[2] - t[1])) # normalize as density
    end

    return p
end

# Exwald pdf at vector of times
function scaled_Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, s::Float64)

    [Exwaldpdf(mu, lambda, s * tau, q) for q in t]

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


# returns closure to compute fractional derivative dq(f, t) = d^q f(t) / dt^q
# using Oustaloop approximation in band (f0, f1) Hz 
function make_fractional_derivative(f::Function, q::Float64, 
         band::Tuple{Float64, Float64} = (.01, 4.0), dt::Float64=DEFAULT_SIMULATION_DT)


    # convert frequency band from Hz to rad/s
    wb = 2.0*band[1]
    wh = 2.0*pi*band[2]

    # Approximation of order 2N+1 (so N=2 is 5th order)
    N = 5
    
    # Compute Oustaloup parameters
    K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
    # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
    residues, _ = oustaloup_residues(K, poles, xeros)
    p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
    M = length(poles)  # ... = 2N+1

    # state: x = [v1, ..., vM]
    x = zeros(Float64,M)
    dx = zeros(Float64,M)

    # fractional derivative
    function dq(t::Float64)

        # Approximate D^q u 
        if (q==0.0) 
            approx_dq = f(t)
        else
            approx_dq = K * f(t)
            for i in 1:M
                approx_dq += residues[i] * x[i]
            end
        end
        
        # internal state update
        for i in 1:M
            x[i] += (-p_i[i] * x[i] + f(t))*dt
        end

        return approx_dq
    end

    # return closure
    return dq 
end


# return trigger threshold for mean interval tau between threshold-crossing events 
# for specified noise Normal(μ,s)
function TriggerThreshold_from_PoissonTau(μ::Float64, s::Float64, τ::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

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

    CV = sqrt(exwaldVariance)/exwaldMean

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

    cvStar = (cv/a)^(1.0/b)

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

using JLD2

"""
Load all .jld2 files and return as vector of named tuples
Each tuple contains filename and data
"""
function process_OU2EXW(folder_path::String)

    # get names of all .jld2 files in folder
    filename = filter(f -> endswith(f, ".jld2"), readdir(folder_path))

    # iniitialize using data from first file
    RawData = load(joinpath(folder_path, filename[1]))
    EXWparam = RawData["EXWparam"]
    Goodness = RawData["Goodness"]
    M, N, P  = size(EXWparam)


    # filter using the remaining files
    for f in 2:length(filename)

        RawData = load(joinpath(folder_path, filename[f]))  

        for i in 1:M 
            for j in 1:N 
                for k in 1:P 
                    if RawData["EXWparam"][i,j,k][3] > EXWparam[i,j,k][3]

                        EXWparam[i,j,k] = RawData["EXWparam"][i,j,k]
                        Goodness[i,j,k] = RawData["Goodness"][i,j,k]

                    end
                end
            end
        end
    end

    # shoulda saved these with output data ... 
    # check the following in demo_OU2Exwald(...)
    mu_o = vec([.001 .002 .005 .01 .02 .05])
    N_mu = length(mu_o)    # = N 
    N_lambda = 8           # = M 
    lambda_o = collect(logrange(1.0e-2, 5.0e1, length = N_lambda))
    N_tau = 32   # = P 
    tau_o = collect(logrange(1.0e-3, 5.0e-2; length=N_tau))

    F = Figure()
    ax = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "LIF μ",
                ylabel = "Exwald μ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
   # ax.xticks = [.005, .01, .02, .05]
  #  ax.title = @sprintf "LIF μ= %.4f" mu_o
    # xlims!(ax, .001, .1)
    # ylims!(ax, .000001, 0.1)

    
    for i in 1:N 
        for j in 1:P 
            lines!(mu_o, [EXWparam[n,i,j][1] for n in 1:M]) 
        end
        #labl = @sprintf "%.4f" lambda_o[j]
  
    end
            display(F)   

    return EXWparam, Goodness, F
end

# linear interpolate y(xx)
# given y(x1) = y1, y(x2) = y2 and x1 < xx < x2
# if x1,x2, y1 or y2 is NaN then return NaN
# unless xx is approximately x1 or x2
function linterp(x::Vector{Float64}, y::Vector{Float64}, xx::Float64)

    if isnan(x[1])
        if isapprox(x[2], xx, rtol = .01)  # nb isapprox(NaN, NaN, ...) is always false
            return y[2]  # possibly NaN
        else 
            return NaN
        end
    elseif isnan(x[2])
        if isapprox(x[1], xx, rtol = .01)  # nb isapprox(NaN, NaN, ...) is always false
            return y[1]  # possibly NaN
        else
            return NaN
        end        
    elseif xx > x[1] && xx < x[2]
        return y[1]*(x[2]-xx)/(x[2]-x[1]) + y[2]*(xx-x[1])/(x[2] - x[1])
    else
        error("Interpolation failed (xx must be between x1 and x2)")
    end

end


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
 