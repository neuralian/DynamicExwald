# Neuralian Toolbox
# MGP August 2024

# splot:  plot spike train
# GLR: Gaussian local rate filter

using Distributions, Plots, ImageFiltering, Sound, Printf, 
        Infiltrator, MLStyle, SpecialFunctions, NLopt, Random

DEFAULT_DT = 1.0e-5
PLOT_SIZE = (800, 600)


# plot spiketime vector as spikes
function splot(spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0)

    N = length(spiketime)
    plot(ones(2, N) .* spiketime', 
        [0.0, height] .* ones(2, N), linewidth = lw, color=:blue, legend=false)
end

function splot!(spiketime::Vector{Float64}, height::Float64=1.0, subplot::Int=1)

    N = length(spiketime)
    plot!(ones(2, N) .* spiketime', [0.0, height] .* ones(2, N), 
        color=:blue, legend=false, subplot = subplot)
end

# Uniform Gaussian rate filter
# Firing rate estimation by sampling Gaussians centred at spike times
#   spiketimes, sd of Gaussian filter, dt = sampling interval, T = end time
# MGP July 2024
# spiketime in seconds
# sd of Gaussian kernal for each spike (if length(sd)==1 then same kernel for all spikes)
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


    print("length(spiketime) = "), println(length(spiketime))
    print("pad= "), println(pad)

    if pad > 0.0
        ifront = findall(spiketime[:] .< pad)[end:-1:1]  # indices of front pad spikes in reverse order (mirror) 
        iback = findall(spiketime[:] .> (T - pad))[end:-1:1]
        #@infiltrate
        spiketime = pad .+ vcat(2.0 * spiketime[1] .- spiketime[ifront[1:end-1]], spiketime, 2.0 * spiketime[iback[1]] .- spiketime[iback[2:end]])
        sd = vcat(sd[ifront[2:end]], sd, sd[iback[2:end]])
    end
    print("length(spiketime) = "), println(length(spiketime))

    # vector to hold result
    padN = Int(ceil(pad / dt))    # pad lengths
    sigN = Int(ceil(T / dt))      # signal length (number of sample points)
    N = sigN + 2 * padN           # padded signal length
    t = (1:N)*dt                  # time vector  (for return)
    r = zeros(N)                  # rate vector


    for i in 1:length(spiketime)
        k = Int(round(spiketime[i] / dt)) # ith spike occurs at this sample point
        n = Int(round(4.0 * sd[i] / dt))  # number of sample points within 4 sd each side of spike
        kernel = pdf(Normal(spiketime[i], sd[i]), k * dt .+ (-n:n) * dt)
        for j in -n:n  # go 4 sd each side
            if (k + j) >= 1 && (k + j) <= N    # in bounds 
                r[k+j] += kernel[n+j+1]
            end
        end
    end

    return ( t, r[padN.+(1:sigN)])


end

# binary Float64 (1.0 or 0.0) vector from spike times 
# nb spike times = cumsum(intervals)
function s2b(spiketime::Vector{Float64}, dt::Float64, T::Float64=ceil(maximum(spiketime)))

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

    audioSampleFreq = 1.0e4
    soundsc(s2b(spiketime, 1.0/audioSampleFreq), audioSampleFreq)     # play audio at 10KHz
    
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
    v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_DT)

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
    v::Function, s::Float64, a::Float64, dt::Float64=DEFAULT_DT)

    #@infiltrate
    x = 0.0
    t0 = 0.0
    t =  0.0
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
    specifiedName::String = "barrier", specifiedValue::Float64 = 1.0)

    @match specifiedName begin   # macro in MLStyle.jl
        "noiseMean" => return(specifiedValue, specifiedValue*mu / sqrt(lambda), specifiedValue*mu)
        "noiseSD"   => return(specifiedValue*sqrt(lambda) / mu, specifiedValue, specifiedValue*sqrt(lambda) )
        "barrier"   => return(specifiedValue / mu, specifiedValue / sqrt(lambda), specifiedValue) 
    end
end

# FPT model (v,s,barrier) defines unique Wald distribution (mu, lambda)
function Wald_parameters_from_FirstpassageTimeModel(v::Float64, s::Float64, barrier::Float64)

    (barrier/v, (barrier/s)^2)

end


# sample of size N from Wald (Inverse Gaussian) distribution via FirstPassageTime_simulate()
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_DT)

    # drift-diffusion model parameters from Wald parameters
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    println(v, " ", s, " ", a)
    FirstPassageTime_simulate(interval, v, s, a, dt)
    return interval
end





#function FirstPassageTime_simulate(interval::Vector{Float64}, v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_DT, T::Float64=1.5*v/a*length(interval))

# Exponential samples by threshold-crossing in Gaussian noise = Normal(m,s)
# NB exponential intervals are ADDED to interval vector for convenience in
#    computing Exwald. If you want exponential intervals, 'interval' must 
#    be zeros when this function is called
function ThresholdTrigger_simulate(interval::Vector{Float64},
    m::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_DT)

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
    m::Function, s::Float64, a::Float64, dt::Float64=DEFAULT_DT)

    t  = 0
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
function Exponential_sample(interval::Vector{Float64}, tau::Float64, dt::Float64=DEFAULT_DT)

    m = 0.0
    s = 1.0
    threshold = TriggerThreshold_from_PoissonTau(m, s, tau, dt)
    # @infiltrate
    ThresholdTrigger_simulate(interval, m, s, threshold, dt)
    (m, s, threshold)  # return trigger mechanism parameters
end

# spike trigger threshold to get mean interval tau between events with noise input Normal(m,s)
function TriggerThreshold_from_PoissonTau(v::Float64, s::Float64, tau::Float64, dt::Float64=DEFAULT_DT)

    quantile(Normal(v, s), 1.0 - dt / tau)  # returns threshold (a)

end

# find mean interval length of threshold trigger with input noise Normal(m,s)
function PoissonTau_from_ThresholdTrigger(m::Float64, s::Float64, threshold::Float64, dt::Float64=DEFAULT_DT)

    dt / (1.0 - cdf(Normal(m, s), threshold))

end


# Exwald samples by simulation
# interval: return vector must be initialized to zeros
#  v:  input noise mean = drift speed
#  s: input noise sd = diffusion coefficient
# barrier: barrier distance for drift-diffusion process
# trigger: trigger threshold for Exponential interval geenration (Poisson process)

function Exwald_simulate(interval::Vector{Float64}, 
            v::Float64, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_DT)

    FirstPassageTime_simulate(interval, v, s, barrier, dt)  # inserts Wald intervals 

    ThresholdTrigger_simulate(interval, v, s, trigger, dt)  # adds Exponential intervals

end

function Exwald_simulate(interval::Vector{Float64}, 
            v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_DT)


    x = 0.0     # drift-diffusion integral 
    t0 = 0.0    # ith interval start time
    t =  0.0    # current time
    i = 1       # interval counter

    #@infiltrate

    while i <= length(interval)  # generate spikes until interval vector is full

        while x < barrier                                   # until reached barrier
            t = t + dt                                      # time update
            x = x + v(t-dt) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
        end                            
        interval[i] = t - t0                        # record time to barrier (Wald sample)
        x = x - barrier                             # reset integral
        t0 = t                                      # next interval start time

        while (v(t-dt) + s * randn()[]) < trigger          # tick until noise crosses trigger level
            t = t + dt
        end
        interval[i] += t - t0                           # add Exponential sample to Wald sample
        t0 = t                                          # next interval start time
        i = i + 1                                       # index for next interval
    end


end



# Exwald samples by simulation
# sample size = length(interval)
function Exwald_sample_sim(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_DT)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    #@infiltrate
    Exwald_simulate(interval, v, s, barrier, trigger, dt)

end

# Exwald sample by sum of inverse Gaussian and Exponential
#function Exwald_sample_sum(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_DT)
function Exwald_sample_sum(N::Int64, mu::Float64, lambda::Float64, tau::Float64, dt::Float64 = DEFAULT_DT)

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

    # drift-diffusion FPT parameters
    # v is Schwarz's μ     (drift)
    # s is Schwarz's σ     (diffusion)
    # a is Schwarz's l     (barrier)
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu,lambda)

    # r = 1.0/tau is Schwarz's λ 
    r = 1.0/tau

    k2 = v^2 - 2.0*r*s^2
    # case 1 (Schwarz p2118)
    if k2 > 0.0

        k = sqrt(k2)
        F =  x-> cdf(Normal(),x)

        #@infiltrate
        return r*exp(-r*t + a*v/s^2)*(exp(-k*a/s^2)*F( (k*t-a)/(s*sqrt(t))) + exp(k*a/s^2)*F(-(k*t + a)/(s*sqrt(t))))

    # case 2
    # nb erfcx(-ix) = w(x)
    else

        k = sqrt(-k2)
        w(x) = erfcx(-1im*x)   # Fadeeva w() function

        #@infiltrate
        return r*exp(-(a-v*t)^2/(2.0*s^2*t))*real( w( k*sqrt(t)/(s*sqrt(2.0)) + 1im*a/(s*sqrt(2.0*t)) ) )

    end

end


# Exwald pdf at vector of times
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64})

    [Exwaldpdf(mu, lambda, tau, s) for s in t]

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
function Exwald_Neuron(N, 
    Exwald_param::Tuple{Float64, Float64, Float64}, 
    stimulus::Function, 
    dt::Float64 = DEFAULT_DT,
    intervals::Bool = false)  # returns spiketimes if false, intervals if true  


    I = zeros(N)    # allocate vector for sample of size N 
  
    dt = DEFAULT_DT  # just to be clear

    # extract Exwald parameters
    (mu, lambda, tau) = Exwald_param

    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)  

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt) 

    # drift rate  = spontaneous + stimulus
    q(t) = v0 + stimulus(t)

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    Exwald_simulate(I, q, s, barrier, trigger, dt)

    # spike train is cumulative sum of intervals
    return intervals ?   I : cumsum(I)

end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
function intervalPhase(spiketime::Vector{Float64}, phase::Float64, freq::Float64, dt::Float64=DEFAULT_DT)

    if phase < 0.0
        phase = 360.0 + phase
    end
    T = maximum(spiketime)
    wavelength = 1.0/freq
    cycles = Int(floor(T/wavelength))  
    sampleTime = wavelength*(phase/360.0 .+ 0:(cycles-1))
    samnpleTime = sampleTime[findall(sampleTime.<T)[1:(end-1)]]  
    interval = zeros(length(sampleTime))

    #@infiltrate
    # interval length at sample times. If sample time is a spike time then get next interval
    if sampleTime[1] < spiketime[1]  # special case: first sample time is before first spike
        interval[1] = spiketime[1]
        i0 = 2
    else
        i0 = 1
    end
    for i in i0:length(sampleTime)
        iBefore = findlast(spiketime .< sampleTime[i])  # index to last spike time before sampleTime[i]
        # if this spike is closer to the sample time than the next spike
        if ( sampleTime[i] - spiketime[iBefore]) < (spiketime[iBefore+1] - sampleTime[i])
            # selected interval ends at spiketime[iBefore] 
            interval[i] = spiketime[iBefore] - spiketime[iBefore-1]
        else
            # seleted interval ends at the following spike
            interval[i] = spiketime[iBefore+1] - spiketime[iBefore]
        end
    end

    return interval

end

# fit Exwald parameters to vector of interspike intervals
#   using maximum likelihood or minimum KL-divergence (Paulin, Pullar and Hoffman, 2024)
# uses NLopt
# returns (mu, lambda, tau)  (in units matching the input data, 
#                             e.g. seconds if intervals are specified in seconds )
function Fit_Exwald_to_ISI(ISI::Vector{Float64}, Pinit::Vector{Float64})

    # likelihood function
    grad = zeros(3)
    LHD = (param, grad) -> sum(log.(Exwaldpdf(param[1], param[2], param[3], ISI))) 

    #@infiltrate

    optStruc = Opt(:LN_PRAXIS,3)   # set up 3-parameter NLopt optimization problem
    
    optStruc.max_objective = LHD       # objective is to maximize likelihood

    optStruc.lower_bounds = [0.0, 0.0, 0.0]   # constrain all parameters > 0
    #optStruc.upper_bounds = [1.0, 25.0,5.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, maxx, ret) = optimize(optStruc, Pinit)

end
