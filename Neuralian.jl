# Neuralian Toolbox
# MGP August 2024

# splot:  plot spike train
# GLR: Gaussian local rate filter

using Distributions, Plots, ImageFiltering, Sound, Printf, Infiltrator

DEFAULT_DT = 1.0e-5
PLOT_SIZE = (800, 600)


# plot spiketime vector as spikes
function splot(spiketime::Vector{Float64}, height::Float64=1.0, lw::Float64=1.0)

    N = length(spiketime)
    plot(ones(2, N) .* spiketime', 
        [0.0, height] .* ones(2, N), linewidth = lw, color=:blue, legend=false)
end

function splot!(spiketime::Vector{Float64}, height::Float64=1.0)

    N = length(spiketime)
    plot!(ones(2, N) .* spiketime', [0.0, height] .* ones(2, N), color=:blue, legend=false)
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



# First passage time simulation parameters for specified Wald parameters with threshold (a) = 1
function FirstPassageTime_parameters_from_Wald(mu::Float64, lambda::Float64)

    return (1.0 / mu, 1.0 / sqrt(lambda), 1.0)  # (v, sigma, alpha) = (drift speed, noise s.d., threshold)

end

# sample of size N from Wald (Inverse Gaussian) distribution via FirstPassageTime_simulate()
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_DT)

    # drift-diffusion model parameters from Wald parameters
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    FirstPassageTime_simulate(interval, v, s, a, dt)
    return interval
end


# Compare first passage time simulation histogram to InverseGaussian from Distributions.jl 
function test_Wald_sample(mu::Float64, lambda::Float64)

    N = 5000  # sample size
    interval = zeros(N)
    Wald_sample(interval, mu, lambda)
    histogram(interval, normalize=:pdf, nbins=100,
        label=@sprintf "First passage times (%.3f, %.3f)" mu lambda)
    t = DEFAULT_DT:DEFAULT_DT:(maximum(interval)*1.2)
    plot!(t, pdf(InverseGaussian(mu, lambda), t),
        label="Wald ($mu, $lambda)", linewidth=2.5, legend=:topleft, size=PLOT_SIZE)
    xlims!(0.0, t[end])
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

# Compare Expsim histogram to Exponential 
function test_Exponential_sample(tau::Float64, dt::Float64=DEFAULT_DT)

    N = 5000  # sample size
    interval = zeros(N)
    (m, s, threshold) = Exponential_sample(interval, tau)
    histogram(interval, normalize=:pdf, nbins=100,
        label=@sprintf "Threshold trigger (%.2f, %.2f, %.2f)" m s threshold)
    t = 0.0:dt:(maximum(interval)*1.2)
    plot!(t, exp.(-t ./ tau) ./ tau, linewidth=2.5, size=PLOT_SIZE,
        label=@sprintf "Exponential ( %.2f)" tau)  #pdf(InverseGaussian(mu, lambda), t))
    xlims!(0.0, t[end])
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

    while i < length(interval)  # generate spikes until interval vector is full

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
function Exwald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_DT)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    #@infiltrate
    Exwald_simulate(interval, v, s, barrier, trigger, dt)

end


# Exwald pdf by convolution of Wald and Exponential pdf
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, t::Vector{Float64})

    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau
    X = imfilter(W, reflect(P))
    X = X / sum(X) / mean(diff(t)) # renormalize (W & P are not normalized because of discrete approx)
    #@infiltrate
end


function test_Exwald_sample(mu, lambda, tau)

    N = 5000  # sample size
    I = zeros(N)
    Exwald_sample(I, mu, lambda, tau)
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau)            # threshold for Poisson component using same noise
    histogram(I, normalize=:pdf, nbins=100, size=(800, 600),
        label=@sprintf "Simulation (%.5f, %.5f, %.5f, %.2f)" v s barrier trigger)

    dt = 1.0e-5
    T = maximum(I) * 1.2
    t = dt:dt:T
    lw = 2.5

    X = Exwaldpdf(mu, lambda, tau, dt, T)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    plot!(t, W, linewidth=lw, size=PLOT_SIZE,
        label=@sprintf "Wald (%.5f, %.5f)" mu lambda)
    plot!(t, P, linewidth=lw, label=@sprintf "Exponential (%.5f)" tau)
    plot!(t, X, linewidth=lw * 1.5, label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)
    ylims!(0.0, max(maximum(X), maximum(W)) * 1.25)
    xlims!(0.0, T)
    #@infiltrate
end

# Exwald spike times 
function Exwald_spiketimes(mu::Float64, lambda::Float64, tau::Float64,
    T::Float64, dt::Float64=1.0e-5)

    # expected number of spikes in time T is T/(mu+tau)                       
    N = Int(ceil(1.5 * T / (mu + tau)))  # probably enough spikes to reach T
    I = zeros(N)
    Exwald_sample(I, mu, lambda, tau, dt)
    spiketime = cumsum(I)
    spiketime = spiketime[findall(spiketime .<= T)]
end


# sample of size N from dynamic Exwald 
# spontaneous Exwald parameters Exwald_param = (mu, lambda, tau)
# stimulus function of time, default f(t)=0.0 (gives spontaneous spike train)
# Default stimulus = 0.0 (spontaneous activity)
function Exwald_Neuron(N, 
    Exwald_param::Tuple{Float64, Float64, Float64}, stimulus::Function, dt::Float64 = DEFAULT_DT)


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
    spiketime = cumsum(I)

end

# return vector of interval lengths in spike train at specified phase (0-360)
#   relative to sin stimulus with frequency freq. 
function intervalsPhase(spiketime::Vector{Float64}, phase::Float64, freq::Float64, dt::Float64=DEFAULT_DT)

    T = maximum(spiketime)
    wavelength = 1.0/freq
    cycles = Int(floor(T/wavelength)) - 1  # actually cycles-1
    sampleTime = (0:cycles)*wavelength*(1.0 + phase/360.0)
    if (sampleTime[end]>T)    # not sure if this can happen but negligible cost to defend
        samnpleTime = sampleTime[1:(end-1)]
    end
    interval = zeros(length(sampleTime))

    @infiltrate
    # interval length at sample times. If sample time is a spike time then get next interval
    for i in 1:length(sampleTime)
        interval[i] = spt[findfirst(spt.>sampleTime[i])] - spt[findlast(spt.<=sampleTime[i])]
    end

    return interval

end

