# Neuralian Toolbox
# Tools for simulating spiking neurons and canal dynamics 
# MGP 2024-25



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

# sample of size N from Wald (Inverse Gaussian) distribution via FirstPassageTime_simulate()
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # drift-diffusion model parameters from Wald parameters
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    println(v, " ", s, " ", a)
    FirstPassageTime_simulate(interval, v, s, a, dt)
    return interval
end


#function FirstPassageTime_simulate(interval::Vector{Float64}, v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_SIMULATION_DT, T::Float64=1.5*v/a*length(interval))

# Samples from Exponential distribution by Gaussian noise = Normal(m,s) threshold crossing
# Generates intervals up to time T=Endpoint if Endpoint is Float64
#   or N = Endpoint intervals if Endpoint is Int64
function exponential_intervals_by_threshold_crossing(m::Float64, s::Float64, Threshold::Float64, 
                        Endpoint::Union{Float64, Int64}, dt::Float64=DEFAULT_SIMULATION_DT)

    if typeof(Endpoint)==Float64
        EndatT = true
        T = Endpoint    # generate intervals until this time
        N = 2*T/PoissonTau_from_ThresholdTrigger(m, s, threshold, dt)      # allocate for 2x expected #spikes 
    else
        EndatT = false
        T = Inf        # keep going until ...
        N = Endpoint   # generate this many intervals
    end

    ISI = zeros(Float64, N)   # to hold intervals

    #@infiltrate
    i = 0
    t = 0.0
    t0 = 0.0
    while true    # will exit when i>N or t>T, which must be finite because tau<infinity
        i += 1
        if i > N                   # have filled ISI, either we are done or need more room
            if EndatT && t < T     # simulation duration not reached, need more room
                M = Int(round(N*.25))
                ISI = append!(ISI, zeros(Float64, M))  # allocate 25% more space
                N = N + M
            else
                return ISI    # done, return N intervals
            end
        end
        while ((m + s * randn()[]) < Threshold)
            t += dt
            if t > T       # reached time T (never happens if EndatT is false because then T = Inf)
                return ISI[1:i-1]   # ith interval has not been generated, number of intervals = i-1
            end
        end
        ISI[i] = t-t0  
        t0 = t
    end
end

# Samples ISI from inhomogenous Exponential distribution by threshold crossing 
#   with time-varying mean rate r(t) = meanrate + Δrate(t) > 0.0
# Generates intervals up to time T=Endpoint if Endpoint is Float64
#   or N = Endpoint intervals if Endpoint is Int64.
# Special case Δrate(t) = 0.0 gives homogenous distribution.
# cumsum(ISI) is event times in a Poisson process with specified (time varying) rate 
function time_varying_exponential_intervals_by_threshold_crossing(meanrate::Float64, Δrate::Function, 
                        Endpoint::Union{Float64, Int64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # trigger level for Poisson process at required mean rate
    # using unit variance Gaussian noise
    v0 = 1.0/meanrate
    s = 1.0
    Threshold = TriggerThreshold_from_PoissonTau(v0, s, 1.0/meanrate, dt)

    if typeof(Endpoint)==Float64
        EndatT = true
        T = Endpoint    # generate intervals until this time
        N = Int(round(2*T/PoissonTau_from_ThresholdTrigger(v0, s, Threshold, dt)))      # allocate for 2x expected #spikes 
    else
        EndatT = false
        T = Inf        # keep going until ...
        N = Endpoint   # generate this many intervals
    end

    ISI = zeros(Float64, N)   # to hold intervals

    i = 0
    t = 0.0
    t0 = 0.0
    τ_sum = 0.0   # for computing mean tau
    while true    # will exit when i>N or t>T, which must be finite because tau<infinity
        i += 1
        if i > N                   # have filled ISI, either we are done or need more room
            if EndatT && t < T     # simulation duration not reached, need more room
                M = Int(round(N*.25))
                ISI = append!(ISI, zeros(Float64, M))  # allocate 25% more space
                N = N + M
            else
                return ISI , τ_sum/N   # done, return N intervals and average rate
            end
        end
        #@infiltrate
        τ = 1.0/(meanrate + Δrate(t))  # mean interval at this instant
        τ_sum += τ
        # find mean noise for expected interval τ (i.e. if noise was fixed until next event)
        v = mean_noise_for_Poisson_given_threshold(τ, s, Threshold, dt)
        while ((v + s * randn()[]) < Threshold)  # until threshold crossed
            t += dt
            if t > T       # reached time T (never happens if EndatT is false because then T = Inf)
                return ISI[1:i-1], τ_sum/i  # ith interval has not been generated, number of intervals = i-1
            end
        end
        ISI[i] = t-t0  
        t0 = t
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

# closure hit = ddstep(t) takes a step in drift-diffusion process 
# with time-varing mean drift speed r(t) = μ + f(t) 
# and variance chosen to match specified IG(μ,λ) (Wald) distribution when f(t)=0.0 ("spontaneous").
# returns true if the particle hit the barrier, and sends the particle back to the start
# otherwise returns false.
function make_drift_diffusion_update(
        Waldparam::Tuple{Float64, Float64}, f::Function, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract parameters of the spontaneous Wald process (f(t)=0.0)  
    (mu, lambda) = Waldparam


    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    x = 0.0  # initial particle location

    function ddstep(t::Float64)

        x = x + (v0 + f(t))*dt + s * randn(1)[] * sqrt(dt)   # integrate noise
        if x >= barrier      
            x = x - barrier       # reset integral for next refractory period
            return true
        else
            return false  # still drifting
        end
    end
 
    return ddstep
end

# first passage time of time varying drift-diffusion to barrier 
# using ddstep = make_drift_diffusion_update().
# Wald/Inverse Gaussian is special case with fixed drift speed
function time_to_barrier(ddstep::Function, t::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    t0 = t
    while !ddstep(t)
        t += dt
    end

    return t-t0
end

# Spike times of "Wald neuron" with time-varying input up to time T
# by simulating first passage time of drift diffusion process 
# with time-varying drift speed v(t) = v0 + f(t)
# and specified spontaneous interval distribution Wald(μ, λ).
function spiketimes_timevaryingWald(Waldparam::Tuple{Float64, Float64}, 
        f::Function, Endpoint::Union{Float64, Int64}, dt = DEFAULT_SIMULATION_DT)

    # unpack Wald parameters
    (mu, lambda) = Waldparam
   
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component

    # time-varying drift speed
    v(t) = v0 + f(t)

    # drift-diffusion update function
    ddstep = make_drift_diffusion_update(Waldparam, f, dt)

    # do we want spikes up to time T, or N spikes?
    # depends on whether Endpoint is a Float64 (time) or Int64 (count)
    # allocate array for spike times
    untilT = typeof(Endpoint)==Float64
    if untilT
        T = Endpoint            # compute spikes up to time T
        N = 2*Int(round(T/mu))  #  2x number of expected spontaneous spikes
    else
        N = Endpoint            # compute N spikes
        T = Inf                 # as long as it takes
    end
    spiketime = zeros(Float64, N)

    t = 0.0
    i = 1
    while  t < T   # end condition depends on spike time or spike count
        interval = time_to_barrier(ddstep, t, dt)
        t = t + interval
        spiketime[i] = t 
        i += 1
        if i>N
            if untilT  # overflow, need room for more spikes
                M = Int(round(N*0.25))  
                append!(spiketime, zeros(Float64, M))
                N = N + M
            else       # we have N spikes
                return spiketime
            end
        end
    end

    return spiketime[1:(i-2)]  # because last recorded spiketime is > T and i has been incremented 
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





        