# Neuralian Toolbox
# MGP August 2024

# splot:  plot spike train
# GLR: Gaussian local rate filter

# Notes 
#   delete!(ax.scene, handle)

using Distributions, GLMakie, ImageFiltering, 
    Sound, PortAudio, SampledSignals, 
    Printf, MLStyle, SpecialFunctions, Random, MAT, 
    BasicInterpolators, DSP

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
    baseline = lines!(ax, [xlims(ax)[1], xlims(ax)[2]], [0.0, 0.0], color = color)
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


   # allocate vector for sample of size N 

    dt = DEFAULT_SIMULATION_DT  # just to be clear

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

# Simulation of torsion pendulum in series with Exwald neuron model, up to time T
# Steinhausen model gives cupula deflection as a function of head angular acceleration,
#   parameterized for chinchilla HSC. 
#   Cupula state x = (p,q) where q = deflection in radians and p=dq/dt, 
#   default initial state x=(0.0,0.0)
# Exwald neuron with input proportional to cupula deflection (~work done on gates)
#   Exwald_param = (mu, lambda, tau)
#   α(t) is head angular acceleration, default α = t->0.0 gives spontaneous spike train.
# 
# MGP Oct 2025
function Steinhausen_Exwald_Neuron(T::Float64, 
    Exwald_param::Tuple{Float64,Float64,Float64},
    α::Function = t->0.0, x::Vector{Float64} = [0.0,0.0],
    dt::Float64=DEFAULT_SIMULATION_DT)  

    dt = DEFAULT_SIMULATION_DT  # just to be clear

    # extract Exwald parameters
    (mu, lambda, tau) = Exwald_param


    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # define a function that returns drift rate of Wald process at time t  
    # drift q(t) = leak  + cupula_deflection(angular_acceleration(t))
    spikeGain = 1000.0  # spikes per second per radian deflection
    q(t) = v0 + spikeGain*cupulaStateUpdate(α(t))[2]  # q is second element of state vector

    # allocate array for spike train, 2x longer than expected length
    Expected_Nspikes = Int(round( T/(mu + tau))) # average number of spikes
    spiketime = zeros(2*Expected_Nspikes)


    # @infiltrate   # << this was active when I was last working on this ... 

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    Exwald_simulate(spiketime, T, q, s, barrier, trigger, dt)


    return spiketime

end

# function to create a closure function 
# that updates cupula state x = (p,q) given angular acceleration wdot (radians/s^2)
# usage: cupulaStateUpdate = create_Steinhausen_state_update(initial_x, dt) # create function & initialize state
#        x = cupulaStateUpdate(wdot) # update state.
#        x[2] is cupula deflection in radians
function create_Steinhausen_state_update(initial_x::Vector{Float64}=[0.0, 0.0], dt::Float64=DEFAULT_SIMULATION_DT)
   
    x = copy(initial_x)  # copy initial state (to avoid mutating the input)

     # Steinhausen model: I.q'' + P.q' + K.q = I.wdot, x = (p,q) where p=dq/dt
    I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
    P = 6.0e-10   # coeff of q', viscous damping N.m.s/rad 
    K = 1.0e-10   # coeff of q, cupula stiffness N.m/rad

    # Euler equation coeffs from model parameters
    PoI = P/I
    KoI = K/I

    function update(wdot::Float64)
    # update (p,q), Euler method
    x1 =  x[1] + (wdot - KoI * x[2] - PoI * x[1]) * dt
    x2 =  x[2] + x[1] * dt
        
    x = [x1, x2]  # Return the updated state vector
    end
    
    return update
end

# function to update cupula state x = (p,q) given angular acceleration wdot (radians/s^2)
cupulaStateUpdate = create_Steinhausen_state_update()



# cupula deflection q (radians) given angular acceleration wdot (radians/s)
# using Steinhausen model 
# returns cupula state variables (q,p) where p = dq/dt
# default initial conditions p0=q0=0.0
function Steinhausen(wdot::Vector{Float64}, dt::Float64=DEFAULT_SIMULATION_DT, x0::Vector{Float64}=[0.0,0.0])

    # Steinhausen model: Iq'' + Pq' + Kq = Awdot, x = (p,q) where p=dq/dt
    # then dp/dt = A/I*wdot - K/I*q - P/I*p

    # initialize state vector
    x = zeros((2,length(wdot)))  # rows are (p,q)
    x[:,1] = x0

    for i in 2:length(wdot)
    
        # update (p,q) 
        x[:,i] = cupulaStateUpdate(wdot[i-1])

    end
    x
end

#############################
#
#    Fractional order model
#
#############################

# Function to compute parameters for Oustaloup approximation to fractional derivative
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
    
    return K, poles, zeros
end

# Function to compute residues for partial fraction decomposition of Oustaloup model
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

# # Closure defining state update function for fractional Steinhausen model y'' + A y' + B Dq y = u(t)
# # with visco-elastic cupular restoring force modeled by fractional derivative Dq = d^q/dt^q
# # Using Oustaloup approximation to Dq over specified frequency band.
# # The augmented state includes auxiliary variables for the Oustaloup approximation.
# # Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
# function make_fractional_Steinhausen_stateUpdate_fcn(
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
#     N = 2
    
#     # Compute Oustaloup parameters
#     K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
#     # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
#     residues, _ = oustaloup_residues(K, poles, xeros)
#     p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
#     M = length(poles)  # ... = 2N+1

#     # Augmented state: u = [y, y', v1, ..., vM]
#     x = [y0; yp0; zeros(M)]
#     du = zeros(length(x))

#     # State update function
#     function update(u::Float64)

#         y =  x[1]
#         dy = x[2]
#         vs = @view x[3:end]
        
#         # Approximate D^q y ≈ K * y + sum r_i * v_i
#         approx_dq = K * y
#         for i in 1:M
#             approx_dq += residues[i] * vs[i]
#         end
        
#         # "ordinary" state updates
#         du[1] = dy
#         du[2] = u - A * dy - B * approx_dq
        
        # Auxiliary state updates
        for i in 1:M
            du[2 + i] = -p_i[i] * vs[i] + y
        end

        # Euler integration
        for i in 1:length(x)
            x[i] += du[i] * dt
        end

        return x[1]  # return cupula deflection
    
    end

    return update
end
    
#############################
#
#    End fractional model
#
#############################

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

# band-limited Gaussian noise by sum-of-sines
#  f0:  lower band limit in Hz 
#  f1:  upper band limit in Hz 
#   s:  Gaussian amplitude s.d.
#  dt:  sample interval
#  Nseconds: Signal duration in seconds
#
# Returns: (t, blg)  time vector and signal vector
# 
function BLG(f0::Float64, f1::Float64, s::Float64, dt::Float64, Nseconds)

    t = collect(0.0:dt:Nseconds)

    bpfilter  = digitalfilter(Bandpass(f0, f1; fs = 1000.), Butterworth(2))

    blg = filt(bpfilter, randn(size(t)))
    blg = s*blg/std(blg)

    (t, blg)

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

function drawBodePlots(freq, Gain, Phase)

    Fig = Figure(size = (800,600))
    #ax1 = Axis(Fig[1,1:2])
    ax2 = Axis(Fig[1,1], xscale = log10, yscale = log10)
    ax3 = Axis(Fig[2,1], xscale = log10, yticks = [-40., -20.0, 0.0, 20., 40.0])
    ylims!(ax3, [-45.0, 45.0])
    
    lines!(ax2, freq, Gain)
    lines!(ax3, freq, Phase)
    
    ylims!(ax2, [0.1, 10.])
    
    display(Fig)
    
   # save("Exwald_Sinewave_Bode.png", Fig)
    
    Fig
end


function sinewave(p, f, t)

    p[1] .+ p[2]*sin.(2.0*π*f*t .- p[3])

end


        