# Neuralian Toolbox
# MGP August 2024

# splot:  plot spike train
# GLR: Gaussian local rate filter

# Notes 
#   delete!(ax.scene, handle)

using Distributions, GLMakie, ImageFiltering, 
    Sound, PortAudio, SampledSignals, 
    Printf, MLStyle, SpecialFunctions, Random, MAT, 
    BasicInterpolators, DSP,
    Infiltrator

DEFAULT_SIMULATION_DT = 1.0e-5
PLOT_SIZE = (800, 600)


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


# sample of size N from Wald (Inverse Gaussian) Distribution
# by simulating first passage times of drift-diffusion to barrier (integrate noise to threshold)
# interval: vector to hold intervals, whose length is the reqd number of intervals
# v: drift rate  
# s: diffusion coeff (std deviation of noise)
# a: barrier height 
# T: simulation time
# dt: simulation time step length, default 1.0e-5 = 10 microseconds
function FirstPassageTime_simulate(interval::Vector{Float64},
    v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    i = 0
    #@infiltrate
    while i < length(interval)
        x = 0.0
        t = 0.0
        while (x < a)
            x = x + v * dt + s * randn(1)[] * sqrt(dt)  # integrate noise
            t += dt
        end
        i = i + 1
        interval[i] = t
    end
    return interval
end

# dynamic version - drift speed v is a function of time
function FirstPassageTime_simulate(interval::Vector{Float64},
    v::Function, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    #@infiltrate
    x = 0.0
    t0 = 0.0
    t = 0.0
    i = 1
    while i < length(interval)
        t = t + dt
        x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)  # integrate noise
        if x > a
            interval[i] = t - t0
            x = x - a
            t0 = t
            i = i + 1
        end
    end
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


# sample of size N from Wald (Inverse Gaussian) distribution via FirstPassageTime_simulate()
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # drift-diffusion model parameters from Wald parameters
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    println(v, " ", s, " ", a)
    FirstPassageTime_simulate(interval, v, s, a, dt)
    return interval
end


#function FirstPassageTime_simulate(interval::Vector{Float64}, v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT, T::Float64=1.5*v/a*length(interval))

# Exponential samples by threshold-crossing in Gaussian noise = Normal(m,s)
# NB exponential intervals are ADDED to interval vector for convenience in
#    computing Exwald. If you want exponential intervals, 'interval' must 
#    be zeros when this function is called
function ThresholdTrigger_simulate(interval::Vector{Float64},
    m::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    i = 0
    N = length(interval)
    #@infiltrate
    while i < length(interval)
        t = 0.0
        while ((m + s * randn()[]) < a)
            t += dt
        end
        i = i + 1
        interval[i] += t # NB adding so we can use the same interval vector for Exp and Exwald (see Exwaldsim)
    end
end

# dynamic version (intervals from inhomogenous Poisson process)
# NB input noise mean is a function 
function ThresholdTrigger_simulate(interval::Vector{Float64},
    m::Function, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    t = 0
    t0 = 0
    count = 1
    while count < length(interval)
        #@infiltrate
        t = t + dt   # force interval > 0
        while (m(t) + s * randn()[]) < a # tick until threshold cross
            t = t + dt
        end
        interval[count] = t - t0
        t0 = t
        count = count + 1
    end
end


# exponential samples by threshold trigger simulation with standard Normal noise
# sample size = length(interval)
function Exponential_sample(interval::Vector{Float64}, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    m = 0.0
    s = 1.0
    threshold = TriggerThreshold_from_PoissonTau(m, s, tau, dt)
    # @infiltrate
    ThresholdTrigger_simulate(interval, m, s, threshold, dt)
    (m, s, threshold)  # return trigger mechanism parameters
end

# spike trigger threshold to get mean interval tau between events with noise input Normal(m,s)
function TriggerThreshold_from_PoissonTau(v::Float64, s::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    quantile(Normal(v, s), 1.0 - dt / tau)  # returns threshold (a)

end

# find mean interval length of threshold trigger with input noise Normal(m,s)
function PoissonTau_from_ThresholdTrigger(m::Float64, s::Float64, threshold::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    dt / (1.0 - cdf(Normal(m, s), threshold))

end


# stationary Exwald samples by simulation
# interval: return vector must be initialized to zeros
#  v:  input noise mean = drift speed
#  s: input noise sd = diffusion coefficient
# barrier: barrier distance for drift-diffusion process
# trigger: trigger threshold for Exponential interval geenration (Poisson process)

# static Exwald by simulation, add 
function Exwald_simulate(interval::Vector{Float64},
    v::Float64, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # nb must call FPT then Poisson sim because FPT replaces, Poisson adds to interval
    FirstPassageTime_simulate(interval, v, s, barrier, dt)  # inserts Wald intervals 

    ThresholdTrigger_simulate(interval, v, s, trigger, dt)  # adds Exponential intervals

end

# Dynamic Exwald simulation in-place
# ExWald_Neuron converts Exwald parameters to physical simulation parameters
#   and then calls this function.
function Exwald_simulate(interval::Vector{Float64},
    v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)


    x = 0.0     # drift-diffusion integral 
    t0 = 0.0    # ith interval start time
    t = 0.0    # current time
    i = 1       # interval counter

    #@infiltrate

    while i <= length(interval)  # generate spikes until interval vector is full

        while x < barrier                                   # until reached barrier                                # time update
            x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            t = t + dt      
        end
        interval[i] = t - t0                        # record time to barrier (Wald sample)
        x = x - barrier                             # reset integral
        t0 = t                                      # next interval start time
  
        while (v(t) + s * randn()[]) < trigger          # tick until noise crosses trigger level
            t = t + dt
        end
        interval[i] += t - t0                           # add Exponential sample to Wald sample
        t0 = t                                          # next interval start time
        i = i + 1                                       # index for next interval
    end

    return interval
end

# simulate dynamic exwald up to time T
# spiketime vector must be large enough to hold the spike train
# if not this function will crash
function Exwald_simulate(spiketime::Vector{Float64}, T::Float64, 
    v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)


    x = 0.0     # drift-diffusion integral 
    t0 = 0.0    # ith interval start time
    t = dt    # current time
    i = 1       # interval counter

    N = length(spiketime)

    #@infiltrate

    spiketime .= 0.0

    while t <= T  # generate spikes until time T

        while x < barrier                                   # until reached barrier                                # time update
            x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            t = t + dt      
        end
        #interval[i] = t - t0                        # record time to barrier (Wald sample)
        x = x - barrier                             # reset integral
        #t0 = t                                      # next interval start time
  
        while (v(t) + s * randn()[]) < trigger          # tick until noise crosses trigger level
            t = t + dt
        end


        spiketime[i] = t                              
        #t0 = t                                          # next interval start time
        i = i + 1                                       # index for next interval
        if i > N
            N = Int(round(1.1* N))
            spiketime = resize!(spiketime, N)   # lengthen spiketime vector by 10%
        end

    end

    N = i-2  # actual length of spike train, last spike time is > T and i has been incremented
    spiketime = resize!(spiketime, N)  # resize to actual length

    # return number of spikes in train
    return N
end

# Dynamic Exwald simulation specifying N intervals
function Exwald_simulate(N::Int64,
    v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

   interval = zeros(N)
   Exwald_simulate(interval, v, s, barrier, trigger, dt)

end

# stationary Exwald samples by simulation
# sample size = length(interval)
function Exwald_sample_sim(N::Int64, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    #@infiltrate
    I = zeros(N)
    Exwald_simulate(I, v, s, barrier, trigger, dt)

    I

end

# Exwald sample by sum of inverse Gaussian and Exponential
#function Exwald_sample_sum(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)
function Exwald_sample_sum(N::Int64, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    rand(InverseGaussian(mu, lambda), N) + rand(Exponential(tau), N)

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

    # force μ, λ, τ > 0.0 
    # for fitting, so Nelder-Mead can go -ve  
    mu = abs(mu)
    lambda = abs(lambda)
    tau = abs(tau)

    if t == 0.0
        return 0.0
    else

        # drift-diffusion FPT parameters
        # v is Schwarz's μ     (drift)
        # s is Schwarz's σ     (diffusion)
        # a is Schwarz's l     (barrier)
        (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)

        # r = 1.0/tau is Schwarz's λ 
        r = 1.0 / tau

        k2 = v^2 - 2.0 * r * s^2
        # case 1 (Schwarz p2118)
        if k2 > 0.0

            k = sqrt(k2)
            F = x -> cdf(Normal(), x)

            #@infiltrate
            return r * exp(-r * t + a * v / s^2) * (exp(-k * a / s^2) * F((k * t - a) / (s * sqrt(t))) + exp(k * a / s^2) * F(-(k * t + a) / (s * sqrt(t))))

            # case 2
            # nb erfcx(-ix) = w(x)
        else

            k = sqrt(-k2)
            w(x) = erfcx(-1im * x)   # Fadeeva w() function

            #@infiltrate
            return r * exp(-(a - v * t)^2 / (2.0 * s^2 * t)) * real(w(k * sqrt(t) / (s * sqrt(2.0)) + 1im * a / (s * sqrt(2.0 * t))))

        end
    end


end


# Exwald pdf at vector of times
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64})

    [Exwaldpdf(mu, lambda, tau, s) for s in t]

end

# Exwald pdf at vector of times
function scaled_Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64}, s::Float64)

    [Exwaldpdf(mu, lambda, s * tau, q) for q in t]

end



# Exwald spike times 
function Exwald_spiketimes(mu::Float64, lambda::Float64, tau::Float64,
    T::Float64, dt::Float64=1.0e-5)

    # expected number of spikes in time T is T/(mu+tau)                       
    N = Int(ceil(1.5 * T / (mu + tau)))  # probably enough spikes to reach T
    I = zeros(N)
    Exwald_sample_sim(I, mu, lambda, tau, dt)
    spiketime = cumsum(I)
    spiketime = spiketime[findall(spiketime .<= T)]
end


# sample of size N from dynamic Exwald 
# spontaneous Exwald parameters Exwald_param = (mu, lambda, tau)
# stimulus function of time, default f(t)=0.0 (gives spontaneous spike train)
# Default stimulus = 0.0 (spontaneous activity)
function Exwald_Neuron_Nspikes(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64},
    stimulus::Function,
    dt::Float64=DEFAULT_SIMULATION_DT,
    intervals::Bool=false)  # returns spiketimes if false, intervals if true  


    I = zeros(N)    # allocate vector for sample of size N 

    dt = DEFAULT_SIMULATION_DT  # just to be clear

    # extract Exwald parameters
    (mu, lambda, tau) = Exwald_param

    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # drift rate  = spontaneous + stimulus
    q(t) = v0 + stimulus(t)

    #@infiltrate

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    Exwald_simulate(I, q, s, barrier, trigger, dt)

    # spike train is cumulative sum of intervals
    return intervals ? I : cumsum(I)

end

# Simulate inhomogenous Exwald for T seconds
# spontaneous Exwald parameters Exwald_param = (mu, lambda, tau)
# stimulus function of time, default f(t)=0.0 (gives spontaneous spike train)
# Default stimulus = 0.0 (spontaneous activity)
function Exwald_Neuron(T::Float64,
    Exwald_param::Tuple{Float64,Float64,Float64},
    stimulus::Function,
    dt::Float64=DEFAULT_SIMULATION_DT)  

    # extract Exwald parameters
    (mu, lambda, tau) = Exwald_param


    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # drift rate  = spontaneous + stimulus
    q(t) = v0 + stimulus(t)

    # allocate array for spike train, 2x longer than expected length
    Expected_Nspikes = Int(round( T/(mu + tau))) # average number of spikes
    spiketime = zeros(2*Expected_Nspikes)


    # @infiltrate   # << this was active when I was last working on this ... 

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    nSpikes = Exwald_simulate(spiketime, T, q, s, barrier, trigger, dt)

    # spike train is cumulative sum of intervals
    return spiketime[1:nSpikes]

end

# returns neuron = (exwald_neuron, EXW_param) 
# where neuron is a closure function to simulate exwald_neuron(δ(t)) given cupula deflection δ(t)
# and EXW_param = (μ, λ, τ) is a tuple containing the neuron's parameters
# closure returns true if the neuron spiked in current time interval, false otherwise
function make_Exwald_neuron(EXW_param::Tuple{Float64, Float64, Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract Exwald parameters
    (mu, lambda, tau) = EXW_param


    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    G = 1.0  # mean input to exwald components is v = v0 + G*δ
    x = 0.0
    refractory = true

    function exwald_neuron(δ::Function, t::Float64)
        
        if refractory 
            x = x + (v0 + G*δ(t))*dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            if x >= barrier      
                refractory = false    # refractory period is over when barrier is reached
                x = x - barrier       # reset integral for next refractory period
            else
                return false  # still refractory
            end
        else
            if (v0 + G*δ(t) + s*randn()[]) > trigger  # Poisson event
                refractory = true
                return true
            else
                return false
            end
        end
    end

    return (exwald_neuron, EXW_param)
end

# returns (spt, input)
# spt = neuron spike train responding to input(t) from time 0 to T 
function spiketimes(neuron::Tuple{Function, Tuple}, input::Function, T::Float64, dt::Float64=DEFAULT_SIMULATION_DT)
    
    # unpack neuron
    (the_neuron, (μ, λ, τ) ) = neuron

    # timesteps
    t = 0.0:dt:T
    M = length(t)

    # record input
    u = zeros(Float64, M)

    # allocate spike time vector 2x expected number of spontaneous spikes
    spontaneous_rate = 1.0/(μ+τ)  
    N = Int(round(2.0*T*spontaneous_rate))
    spiketime = zeros(Float64, N)

    j = 0 # spike count
    for i in 1:M
        u[i] = input(t[i])
        if the_neuron(input, t[i])==true   # neuron fired
            j += 1 
            if i > N   # overflow
                N = Int(round(N*1.25))
                resize!(spiketime, N)  # make spiketime vector 25% longer
            end
            spiketime[j] = t[i]
        end
    end

    return (spiketime[1:j], u)
end

# returns N interspike intervals from neuron responding to input(t) from time 0 to T 
function interspike_intervals(neuron::Tuple{Function, Tuple}, input::Function, N::Int64, dt::Float64=DEFAULT_SIMULATION_DT)
    
    # unpack neuron
    (the_neuron, (μ, λ, τ) ) = neuron

    interval = zeros(Float64, N)

    t = 0.0
    previous_spiketime = 0.0
    i = 0
    while i < N
        t += dt
        if the_neuron(input(t))==true   # neuron fired
            i += 1 
            interval[i] = t - previous_spiketime
            previous_spiketime = t  
        end
    end

    return interval
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

# Create closure to compute state update for fractional Steinhausen model 
#       y'' + A y' + B Dq y = Dq u(t)
# with fractional derivative Dq = d^q/dt^q, -1<q<1 (Landolt and Correia, 1980; Magin, 2005).
#   q = 0.0 gives classical torsion pendulum model
#   q > 0.0 gives phase advance πq/2 at all frequencies, 
#           in particular q=1.0 turns velocity sensitivity into acceleration sensitivity.
# Accurate enough in the specified frequency band (f0, f1) /Hz. Outside that all bets are off.
# Construct the state update function and initialize state: 
#       cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(<params>)
# Use the state update function to update state at ith timestep and return cupula deflection (rad):
#       cupula_deflection = cupula_update(u_i)
# Uses Oustaloup approximation to Dq over specified frequency band (f0,f1) in Hz.
# Default order of approximation N=2 (M=2*N+1 = 5th order linear TF), use N up to 5 for better approx.
# Augmented state includes auxiliary variables for the approximation.
# Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
# Example usage
        # q = -0.5  # fractional order
        # w = 2.0  # frequency of input rad/s
        # T = 12.0
        # dt = DEFAULT_SIMULATION_DT
        # t = 0:dt:T
        # x = sin.(w .* t)  # input angular velocity (rad/s)
        # d = zeros(length(t))

        # FSS_update = make_fractional_Steinhausen_stateUpdate_fcn(q, 0., 0.)

        # for i in 1:length(t)
        #     d[i] = FSS_update(x[i])
        # end
##
function make_fractional_Steinhausen_stateUpdate_velocity_fcn(
    q::Float64, y0::Float64, yp0::Float64; 
    dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

    # Fractional Steinhausen model: I.y'' + P.y' + K.Dq y = I.wdot, 
    I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
    P = 6.0e-11   # coeff of q', viscous damping N.m.s/rad 
    G = 1.0e-10    # coeff of q, cupula stiffness N.m/rad 

    # Update equation coeffs from model parameters
    A = P/I
    B = G/I
 
    # convert frequency band from Hz to rad/s
    wb = 2.0*pi*f0
    wh = 2.0*pi*f1

    # Approximation of order 2N+1 (so N=2 is 5th order)
    N = 5
    
    # Compute Oustaloup parameters
    K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
    # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
    residues, _ = oustaloup_residues(K, poles, xeros)
    p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
    M = length(poles)  # ... = 2N+1

    # Augmented state: x = [y, y', v1, ..., vM]
    x = [y0; yp0; zeros(M)]
    du = zeros(length(x))

    # State update function of angular velocity ̇ω at current time step
    function update(ω::Function, t::Float64)

        wdot = diffcd(ω, t)  # angular acceleration

        y =  x[1]
        dy = x[2]
        vs = @view x[3:end]
        
        # Approximate D^q u 
        if (q==0.0) 
            approx_dq = wdot
        else
            approx_dq = K * wdot
            for i in 1:M
                approx_dq += residues[i] * vs[i]
            end
        end
        
        # "ordinary" state updates
        du[1] = dy
        du[2] = approx_dq - A * dy - B * y
        
        # Auxiliary state updates
        for i in 1:M
            du[2 + i] = -p_i[i] * vs[i] + wdot
        end

        # Euler integration
        for i in 1:length(x)
            x[i] += du[i] * dt
        end

        return x[1]  # return cupula deflection
    
    end

    # return closure
    return update 
end


# # Create closure to compute state update for fractional Steinhausen model 
# #       y'' + A y' + B Dq y = Dq u(t)
# # with fractional derivative Dq = d^q/dt^q, -1<q<1 (Landolt and Correia, 1980; Magin, 2005).
# #   q = 0.0 gives classical torsion pendulum model
# #   q > 0.0 gives phase advance πq/2 at all frequencies, 
# #           in particular q=1.0 turns velocity sensitivity into acceleration sensitivity.
# # Accurate enough in the specified frequency band (f0, f1) /Hz. Outside that all bets are off.
# # Construct the state update function and initialize state: 
# #       cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(<params>)
# # Use the state update function to update state at ith timestep and return cupula deflection (rad):
# #       cupula_deflection = cupula_update(u_i)
# # Uses Oustaloup approximation to Dq over specified frequency band (f0,f1) in Hz.
# # Default order of approximation N=2 (M=2*N+1 = 5th order linear TF), use N up to 5 for better approx.
# # Augmented state includes auxiliary variables for the approximation.
# # Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
# # Example usage
#         # q = -0.5  # fractional order
#         # w = 2.0  # frequency of input rad/s
#         # T = 12.0
#         # dt = DEFAULT_SIMULATION_DT
#         # t = 0:dt:T
#         # x = sin.(w .* t)  # input angular velocity (rad/s)
#         # d = zeros(length(t))

#         # FSS_update = make_fractional_Steinhausen_stateUpdate_fcn(q, 0., 0.)

#         # for i in 1:length(t)
#         #     d[i] = FSS_update(x[i])
#         # end
# ##
# function make_fractional_Steinhausen_stateUpdate_velocity_fcn(
#     q::Float64, y0::Float64, yp0::Float64; 
#     dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

#     # Fractional Steinhausen model: I.y'' + P.y' + K.Dq y = I.wdot, 
#     I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
#     P = 6.0e-11   # coeff of q', viscous damping N.m.s/rad 
#     G = 1.0e-10    # coeff of q, cupula stiffness N.m/rad 

#     # Update equation coeffs from model parameters
#     A = P/I
#     B = G/I
 
#     # convert frequency band from Hz to rad/s
#     wb = 2.0*pi*f0
#     wh = 2.0*pi*f1

#     # Approximation of order 2N+1 (so N=2 is 5th order)
#     N = 5
    
#     # Compute Oustaloup parameters
#     K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
#     # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
#     residues, _ = oustaloup_residues(K, poles, xeros)
#     p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
#     M = length(poles)  # ... = 2N+1

#     # Augmented state: x = [y, y', v1, ..., vM]
#     x = [y0; yp0; zeros(M)]
#     du = zeros(length(x))

#     # State update function of angular velocity ̇ω at current time step
#     function update(ω::Function, t::Float64)

#         wdot = diffcd(ω, t)  # angular acceleration

#         y =  x[1]
#         dy = x[2]
#         vs = @view x[3:end]
        
#         # Approximate D^q u 
#         if (q==0.0) 
#             approx_dq = wdot
#         else
#             approx_dq = K * wdot
#             for i in 1:M
#                 approx_dq += residues[i] * vs[i]
#             end
#         end
        
#         # "ordinary" state updates
#         du[1] = dy
#         du[2] = approx_dq - A * dy - B * y
        
#         # Auxiliary state updates
#         for i in 1:M
#             du[2 + i] = -p_i[i] * vs[i] + wdot
#         end

#         # Euler integration
#         for i in 1:length(x)
#             x[i] += du[i] * dt
#         end

#         return x[1]  # return cupula deflection
    
#     end

#     # return closure
#     return update 
# end


# Create closure to compute state update for fractional Steinhausen model 
#       y'' + A y' + B Dq y = Dq u(t)
# with fractional derivative Dq = d^q/dt^q, -1<q<1 (Landolt and Correia, 1980; Magin, 2005).
#   q = 0.0 gives classical torsion pendulum model
#   q > 0.0 gives phase advance πq/2 at all frequencies, 
#           in particular q=1.0 turns velocity sensitivity into acceleration sensitivity.
# Accurate enough in the specified frequency band (f0, f1) /Hz. Outside that all bets are off.
# Construct the state update function and initialize state: 
#       cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(<params>)
# Use the state update function to update state at ith timestep and return cupula deflection (rad):
#       cupula_deflection = cupula_update(u_i)
# Uses Oustaloup approximation to Dq over specified frequency band (f0,f1) in Hz.
# Default order of approximation N=2 (M=2*N+1 = 5th order linear TF), use N up to 5 for better approx.
# Augmented state includes auxiliary variables for the approximation.
# Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
# Example usage
        # q = -0.5  # fractional order
        # w = 2.0  # frequency of input rad/s
        # T = 12.0
        # dt = DEFAULT_SIMULATION_DT
        # t = 0:dt:T
        # x = sin.(w .* t)  # input angular velocity (rad/s)
        # d = zeros(length(t))

        # FSS_update = make_fractional_Steinhausen_stateUpdate_fcn(q, 0., 0.)

        # for i in 1:length(t)
        #     d[i] = FSS_update(x[i])
        # end
##
# function make_fractional_Steinhausen_stateUpdate_velocity_fcn(
#     q::Float64, y0::Float64, yp0::Float64; 
#     dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

#     # Fractional Steinhausen model: I.y'' + P.y' + K.Dq y = I.wdot, 
#     I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
#     P = 6.0e-11   # coeff of q', viscous damping N.m.s/rad 
#     G = 1.0e-10    # coeff of q, cupula stiffness N.m/rad 

#     # Update equation coeffs from model parameters
#     A = P/I
#     B = G/I
 
#     # convert frequency band from Hz to rad/s
#     wb = 2.0*pi*f0
#     wh = 2.0*pi*f1

#     # Approximation of order 2N+1 (so N=2 is 5th order)
#     N = 5
    
#     # Compute Oustaloup parameters
#     K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
#     # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
#     residues, _ = oustaloup_residues(K, poles, xeros)
#     p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
#     M = length(poles)  # ... = 2N+1

#     # Augmented state: x = [y, y', v1, ..., vM]
#     x = [y0; yp0; zeros(M)]
#     du = zeros(length(x))

#     # State update function of angular velocity ̇ω at current time step
#     function update(ω::Function, t::Float64)

#         wdot = diffcd(ω, t)  # angular acceleration

#         y =  x[1]
#         dy = x[2]
#         vs = @view x[3:end]
        
#         # Approximate D^q u 
#         if (q==0.0) 
#             approx_dq = wdot
#         else
#             approx_dq = K * wdot
#             for i in 1:M
#                 approx_dq += residues[i] * vs[i]
#             end
#         end
        
#         # "ordinary" state updates
#         du[1] = dy
#         du[2] = approx_dq - A * dy - B * y
        
#         # Auxiliary state updates
#         for i in 1:M
#             du[2 + i] = -p_i[i] * vs[i] + wdot
#         end

#         # Euler integration
#         for i in 1:length(x)
#             x[i] += du[i] * dt
#         end

#         return x[1]  # return cupula deflection
    
#     end

#     # return closure
#     return update 
# end


# Create closure to compute state update for fractional Steinhausen model 
#       y'' + A y' + B Dq y = Dq u(t)
# with fractional derivative Dq = d^q/dt^q, -1<q<1 (Landolt and Correia, 1980; Magin, 2005).
#   q = 0.0 gives classical torsion pendulum model
#   q > 0.0 gives phase advance πq/2 at all frequencies, 
#           in particular q=1.0 turns velocity sensitivity into acceleration sensitivity.
# Accurate enough in the specified frequency band (f0, f1) /Hz. Outside that all bets are off.
# Construct the state update function and initialize state: 
#       cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(<params>)
# Use the state update function to update state at ith timestep and return cupula deflection (rad):
#       cupula_deflection = cupula_update(u_i)
# Uses Oustaloup approximation to Dq over specified frequency band (f0,f1) in Hz.
# Default order of approximation N=2 (M=2*N+1 = 5th order linear TF), use N up to 5 for better approx.
# Augmented state includes auxiliary variables for the approximation.
# Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
# Example usage
        # q = -0.5  # fractional order
        # w = 2.0  # frequency of input rad/s
        # T = 12.0
        # dt = DEFAULT_SIMULATION_DT
        # t = 0:dt:T
        # x = sin.(w .* t)  # input angular acceleration (rad/s)
        # d = zeros(length(t))

        # FSS_update = make_fractional_Steinhausen_stateUpdate_fcn(q, 0., 0.)

        # for i in 1:length(t)
        #     d[i] = FSS_update(x[i])
        # end
##
function make_fractional_Steinhausen_stateUpdate_acceleration_fcn(
    q::Float64, y0::Float64, yp0::Float64; 
    dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

    # Fractional Steinhausen model: I.y'' + P.y' + K.Dq y = I.wdot, 
    I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
    P = 6.0e-11   # coeff of q', viscous damping N.m.s/rad 
    G = 1.0e-10    # coeff of q, cupula stiffness N.m/rad 

    # Update equation coeffs from model parameters
    A = P/I
    B = G/I
 
    # convert frequency band from Hz to rad/s
    wb = 2.0*pi*f0
    wh = 2.0*pi*f1

    # Approximation of order 2N+1 (so N=2 is 5th order)
    N = 5
    
    # Compute Oustaloup parameters
    K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
    # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
    residues, _ = oustaloup_residues(K, poles, xeros)
    p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
    M = length(poles)  # ... = 2N+1

    # Augmented state: x = [y, y', v1, ..., vM]
    x = [y0; yp0; zeros(M)]
    du = zeros(length(x))

    # State update function of angular acceleration ̇ω at current time step
    function update(ωdot::Function, t::Float64)

        wdot = ωdot(t)

        y =  x[1]
        dy = x[2]
        vs = @view x[3:end]
        
        # Approximate D^q u 
        if (q==0.0) 
            approx_dq = wdot
        else
            approx_dq = K * wdot
            for i in 1:M
                approx_dq += residues[i] * vs[i]
            end
        end
        
        # "ordinary" state updates
        du[1] = dy
        du[2] = approx_dq - A * dy - B * y
        
        # Auxiliary state updates
        for i in 1:M
            du[2 + i] = -p_i[i] * vs[i] + wdot
        end

        # Euler integration
        for i in 1:length(x)
            x[i] += du[i] * dt
        end

        return x[1]  # return cupula deflection
    
    end

    # return closure
    return update
end

    
# Spike times of Exwald neuron with input proportional to cupula deflection (δ)
# computed by fractional torsion pendulum model, δ'' + Aδ' + Bδ = Dq D w(t)
#   EXW_param = (μ, λ, τ)
#   q=0 gives classical torsion pendulum, 0.0<q<1 gives frequency-independent phase advance πq/2
#   Head angular velocity is a function of t on simulation interval (0,T)
#   e.g. w =  t->sinewave((0.0, 1.0, 0.0), 2.0*pi, t) # 1Hz unit amplitude sine wave
#   w must be defined at t-dt and t+dt for every t, so we can compute central difference derivative
# Returns vector of spike times
function fractionalSteinhausenExwald_Neuron(q::Float64, EXW_param::Tuple{Float64, Float64, Float64},
    w::Function, T::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract Exwald parameters (for clarity)
    (mu, lambda, tau) = EXW_param

    # angular acceleration wdot from w
    wdot = t-> diffcd(w, t, dt)

    # Parameters of drift-diffusion process time-to-barrier model corresponding to Wald component   
    # v0 = spontaneous drift rate, s = noise amplitude (diffusion rate), barrier = barrier distance
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # Trigger level for threshold-crossing times of N(v0,s) noise to be a Poisson process
    # with mean interval tau.
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # allocate array for spike train, 2x longer than expected number of spontaneous spikes up to time T
    # It will be extended later if it turns out not to be big enough
    N = 2*Int(ceil( T/(mu + tau)))
    spiketime = zeros(Float64, N)

    # initial canal state
    δ    = 0.0
    δdot = 0.0

    # instantiate and initialize torsion pendulum model 
    cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(q, δ, δdot)

    # This function specifies input to the Exwald process (Exwald neuron responding to head acceleration)
    # Head angular acceleration wdot(t) determines cupula deflection δ(t) 
    # which affects the drift rate of the Wald process and the threshold-crosssing rate of the Poisson.
    # Note that cupula_update() is a dynamical filter, it has internal state variables 
    # such that its output δ(t) is a function of state and input as per fractional torsion pendulum model.
    # TBD set gain parameter G to match firing rate gains (spikes/s per deg/s) to data
    # for neuron with the specified dynamic parameters.  
    # e.g. see Landolt and Correia 1980
    G = 0.5
    v(t) = v0 + G*cupula_update(wdot(t))

    t = 0.0    # current time
    x = 0.0    # current location of drift-diffusion particle
    i = 0      # index for inserting spike times into spiketime vector
    while t <= T  

        while x < barrier                                   # until reached barrier                                # time update
            x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            t = t + dt      
        end
        # x has hit the barrier, t - spiketime[i-1] is a sample from Wald(mu,lambda) 

        # reset the drift-diffusion process
        # nb we could interpolate the exact time-to-barrier and set x = 0.0
        # but dt is small enough to get t close enough
        x = x - barrier                           
   
        # wait for Poisson event 
        while (v(t) + s * randn()[]) < trigger          
            t = t + dt
        end
        # Poisson event generated, t - spiketime[i-1] is sample from Exwald(mu, lambda, tau)

        # update spiketime index, make spiketime 25% longer if we have run off the end
        i += 1
        if i > N
            N = Int(round(1.25* N))
            spiketime = resize!(spiketime, N)   # lengthen spiketime vector by 10%
        end

        # put spike time in spiketime
        spiketime[i] = t                              

    end

    return spiketime[1:i]
end
   
# N spike times of Exwald neuron with input proportional to cupula deflection (δ)
# computed by fractional torsion pendulum model with angular velocity input w(t), 
#         δ'' + Aδ' + Bδ = Dq D w(t)
#   EXW_param = (μ, λ, τ)
#   q=0 gives classical torsion pendulum, 0.0<q<1 gives frequency-independent phase advance πq/2
#   Input angular velocity e.g. w = t->sinewave((0.0, 1.0, 0.0), 1.0, t) # w(t) = sin(2πt)
#      w must be able to return w(t-dt) and w(t+dt) if it is called at t, 
#         because its derivative (angular acceleration) will be computed by central difference.
#         So if w(t) is implemented using a vector of function values (ie w(t)is an alias for w[t])
#         you (may) need special cases for w(-dt) and w(T+dt) to compute the derivative at endpoints.  
# Returns vector of spike times
function fractionalSteinhausenExwald_Neuron(q::Float64, EXW_param::Tuple{Float64, Float64, Float64},
    w::Function, N::Int64, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract Exwald parameters (for clarity)
    (mu, lambda, tau) = EXW_param

    # wdot(t) is first derivative of w(.) at t (angular acceleration given angular velocity)
    wdot = t -> diffcd(w, t)

    # Parameters of drift-diffusion process time-to-barrier model corresponding to Wald component   
    # v0 = spontaneous drift rate, s = noise amplitude (diffusion rate), barrier = barrier distance
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # Trigger level for threshold-crossing times of N(v0,s) noise to be a Poisson process
    # with mean interval tau.
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # allocate array for spike train, 
    spiketime = zeros(Float64, N)

    # initial canal state
    δ    = 0.0
    δdot = 0.0

    # Instantiate and initialize torsion pendulum model 
    # cupula_update(wdot(t)) returns cupula deflection (radians) given head angular acceleration at t 
    # This function has internal state variables (δ, δ') plus M=2N+1 auxiliary state variables
    #      for an Mth-order approximation to the fractional derivative   
    cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(q, δ, δdot)

    # Specify the time-varying noise input to the physical components of the Exwald process
    # In the (simplest) model implemented here, cupula deflection δ(t) depends dynamically on 
    #   head angular acceleration via the fractional Steinhausen model 
    #   (using parameters from Hullar and Correia and Landolt)  
    #   Spontaneous discharge is modelled using gaussian noise crossing a trigger threshold to generate
    #   Poisson events and a drift-diffusion process to model the Wald censoring process.
    #   I assume that its the same Gaussian noise N(m, s) = m + N(0,s) in each sub-process, and that
    #   cupula deflection linearly alters the mean. This has a simple physical interpretation ie
    #   the noise is (noisy) net current entering a hair cell whose mean rises and falls with cupula deflection.  
    #   TBD set gain parameter G to match firing rate gains (spikes/s per deg/s) to data
    #   for neuron with the specified dynamic parameters.  I have picked a value for G here that gives plausible
    #   results but probably this needs to be a neuron-specific number passed as an argument.
    #   see Landolt and Correia 1980
    G = 0.5
    v(t) =  v0 + G*cupula_update(wdot(t))


    t = 0.0    # current time
    x = 0.0    # current location of drift-diffusion particle
    i = 0      # index for inserting spike times into spiketime vector
    while i < N

        while x < barrier                                   # until reached barrier                                # time update
            x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            t = t + dt      
        end
        # x has hit the barrier, t - spiketime[i-1] is a sample from Wald(mu,lambda) 

        # reset the drift-diffusion process
        # nb we could interpolate the exact time-to-barrier and set x = 0.0
        # but dt is small enough to get t close enough
        x = x - barrier                           
   
        # wait for Poisson event 
        while (v(t) + s * randn()[]) < trigger          
            t = t + dt
        end
        # Poisson event generated, t - spiketime[i-1] is sample from Exwald(mu, lambda, tau)

        # update spiketime index, make spiketime 25% longer if we have run off the end
        i += 1
        if i > N
            N = Int(round(1.25* N))
            spiketime = resize!(spiketime, N)   # lengthen spiketime vector by 10%
        end

        # put spike time in spiketime
        spiketime[i] = t                              

    end

    return spiketime
end



# return closure fsx(̇ω) that simulates fractional Steinhausen-Exwald neuron 
# responding to head angular acceleration ̇ω(t) (NB blg() returns (ω, ̇ω))
# returns 1.0 if the neuron spikes at time t, otherwise 0.0
# nb must be called at fixed rate (interval dt) 
function make_fractionalSteinhausenExwald_Neuron(q::Float64, EXW_param::Tuple{Float64, Float64, Float64},
   dt::Float64=DEFAULT_SIMULATION_DT)

    # extract Exwald parameters (for clarity)
    (mu, lambda, tau) = EXW_param

    # Parameters of drift-diffusion process time-to-barrier model corresponding to Wald component   
    # v0 = spontaneous drift rate, s = noise amplitude (diffusion rate), barrier = barrier distance
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # Trigger level for threshold-crossing times of N(v0,s) noise to be a Poisson process
    # with mean interval tau.
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # initial canal state
    δ    = 0.0
    δdot = 0.0

    # instantiate and initialize torsion pendulum model 
    cupula_update = make_fractional_Steinhausen_stateUpdate_fcn(q, δ, δdot)

    # Gain G specifies input to Exwald neuron in response to head acceleration wdot
    # TBD set gain parameter G to match firing rate gains (spikes/s per deg/s) to data
    # for neuron with the specified dynamic parameters.  
    # e.g. see Landolt and Correia 1980
    G = 0.5
    v(wdot) = v0 + G*cupula_update(wdot)

    t = 0.0    # current time
    x = 0.0    # current location of drift-diffusion particle
    i = 0      # index for inserting spike times into spiketime vector

    refractory = true

    function fsx(wdot::Float64)

        if refractory 
            x += x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            if x >= barrier
                refractory = false
                x -= barrier  # reset integral ready for next refractory period
            end
            return 0.0
        else
            if (v(t) + s * randn()[]) >= trigger 
                refractory = true   # start new refractory period
                return 1.0
            else
                return 0.0
            end
        end
    end

    return fsx  # return closure.  fsx(wdot) returns 1 if spike occurs 

end

# # intervals generated by fractional steinhausen Exwald neuron 
# # responding to angular acceleration wdot(t) up to time T
# # nb spike times = cumsum(intervals)
# function fsx_intervals(fsx, wdot::Function, T::Float64, dt::Float64 = DEFAULT_SIMULATION_DT)

#     t=0.0
#     Δt = 0.0
#     while t<T
#         t + Δt
#         if (fsx(wdot(t)))  

 
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
    return (ms2s*mu_0, ms2s*10.0^log_lambda, ms2s*10.0^log_tau)

end

function Exwald_fromCVStar(CVStar::Float64, tbar::Float64)
#TBD

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



        