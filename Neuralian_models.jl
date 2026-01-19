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

# Intervals ISI drawn from inhomogenous Exponential distribution by Gaussian noise threshold crossing 
#   with time-varying mean rate r(t) = baserate + Δrate(t) > 0.0
# Simulate to T=Endpoint if Endpoint is Float64 or generate N=Endpoint intervals if Endpoint is Int64.
# Special case Δrate(t) = 0.0 gives homogenous distribution (constant mean rate = baserate)
# cumsum(ISI) is event times in a Poisson process with specified (time varying) rate 
function time_varying_exponential_intervals_by_threshold_crossing(baserate::Float64, Δrate::Function, 
                        Endpoint::Union{Float64, Int64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # trigger level for Poisson process at required mean rate
    # using unit variance Gaussian noise
    v0 = 1.0/baserate
    s = 1.0
    Threshold = TriggerThreshold_from_PoissonTau(v0, s, 1.0/baserate, dt)

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
        τ = 1.0/(baserate + Δrate(t))  # mean interval at this instant
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

# closure to generate pink noise step dP
# by adding N white noise samples (Voss pink noise generator)
function make_pink_noise(N=24)

    count = 0
    whites = randn(N)   # samples of white noise
    P = 0.0

    function pink()

        count += 1
        ntrail = trailing_zeros(count)  # number of trailing zeros in binary representation of count
        i = (ntrail % N) + 1            # update row i every 2^i steps 

        new_white = randn()
        P += (new_white - whites[i])    
        whites[i] = new_white

        return P / sqrt(N)

    end

    return pink

end




# # stationary Exwald samples by simulation
# # interval: return vector must be initialized to zeros
# #  v:  input noise mean = drift speed
# #  s: input noise sd = diffusion coefficient
# # barrier: barrier distance for drift-diffusion process
# # trigger: trigger threshold for Exponential interval geenration (Poisson process)

# # static Exwald by simulation, add 
# function Exwald_simulate(interval::Vector{Float64},
#     v::Float64, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

#     # nb must call FPT then Poisson sim because FPT replaces, Poisson adds to interval
#     FirstPassageTime_simulate(interval, v, s, barrier, dt)  # inserts Wald intervals 

#     ThresholdTrigger_simulate(interval, v, s, trigger, dt)  # adds Exponential intervals

# end

# # Dynamic Exwald simulation in-place
# # ExWald_Neuron converts Exwald parameters to physical simulation parameters
# #   and then calls this function.
# function Exwald_simulate(interval::Vector{Float64},
#     v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)


#     x = 0.0     # drift-diffusion integral 
#     t0 = 0.0    # ith interval start time
#     t = 0.0    # current time
#     i = 1       # interval counter

#     #@infiltrate

#     while i <= length(interval)  # generate spikes until interval vector is full

#         while x < barrier                                   # until reached barrier                                # time update
#             x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
#             t = t + dt      
#         end
#         interval[i] = t - t0                        # record time to barrier (Wald sample)
#         x = x - barrier                             # reset integral
#         t0 = t                                      # next interval start time
  
#         while (v(t) + s * randn()[]) < trigger          # tick until noise crosses trigger level
#             t = t + dt
#         end
#         interval[i] += t - t0                           # add Exponential sample to Wald sample
#         t0 = t                                          # next interval start time
#         i = i + 1                                       # index for next interval
#     end

#     return interval
# end

# # simulate dynamic exwald up to time T
# # spiketime vector must be large enough to hold the spike train
# # if not this function will crash
# function Exwald_simulate(spiketime::Vector{Float64}, T::Float64, 
#     v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)


#     x = 0.0     # drift-diffusion integral 
#     t0 = 0.0    # ith interval start time
#     t = dt    # current time
#     i = 1       # interval counter

#     N = length(spiketime)

#     #@infiltrate

#     spiketime .= 0.0

#     while t <= T  # generate spikes until time T

#         while x < barrier                                   # until reached barrier                                # time update
#             x = x + v(t) * dt + s * randn(1)[] * sqrt(dt)   # integrate noise
#             t = t + dt      
#         end
#         #interval[i] = t - t0                        # record time to barrier (Wald sample)
#         x = x - barrier                             # reset integral
#         #t0 = t                                      # next interval start time
  
#         while (v(t) + s * randn()[]) < trigger          # tick until noise crosses trigger level
#             t = t + dt
#         end


#         spiketime[i] = t                              
#         #t0 = t                                          # next interval start time
#         i = i + 1                                       # index for next interval
#         if i > N
#             N = Int(round(1.1* N))
#             spiketime = resize!(spiketime, N)   # lengthen spiketime vector by 10%
#         end

#     end

#     N = i-2  # actual length of spike train, last spike time is > T and i has been incremented
#     spiketime = resize!(spiketime, N)  # resize to actual length

#     # return number of spikes in train
#     return N
# end

# # Dynamic Exwald simulation specifying N intervals
# function Exwald_simulate(N::Int64,
#     v::Function, s::Float64, barrier::Float64, trigger::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

#    interval = zeros(N)
#    Exwald_simulate(interval, v, s, barrier, trigger, dt)

# end

# # stationary Exwald samples by simulation
# # sample size = length(interval)
# function Exwald_sample_sim(N::Int64, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

#     (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
#     trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

#     #@infiltrate
#     I = zeros(N)
#     Exwald_simulate(I, v, s, barrier, trigger, dt)

#     I

# end

# # Exwald sample by sum of inverse Gaussian and Exponential
# #function Exwald_sample_sum(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)
# function Exwald_sample_sum(N::Int64, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

#     rand(InverseGaussian(mu, lambda), N) + rand(Exponential(tau), N)

# end


# # Exwald spike times 
# function Exwald_spiketimes(mu::Float64, lambda::Float64, tau::Float64,
#     T::Float64, dt::Float64=1.0e-5)

#     # expected number of spikes in time T is T/(mu+tau)                       
#     N = Int(ceil(1.5 * T / (mu + tau)))  # probably enough spikes to reach T
#     I = zeros(N)
#     Exwald_sample_sim(I, mu, lambda, tau, dt)
#     spiketime = cumsum(I)
#     spiketime = spiketime[findall(spiketime .<= T)]
# end


# # sample of size N from dynamic Exwald 
# # spontaneous Exwald parameters Exwald_param = (mu, lambda, tau)
# # stimulus function of time, default f(t)=0.0 (gives spontaneous spike train)
# # Default stimulus = 0.0 (spontaneous activity)
# function Exwald_Neuron_Nspikes(N::Int64,
#     Exwald_param::Tuple{Float64,Float64,Float64},
#     stimulus::Function,
#     dt::Float64=DEFAULT_SIMULATION_DT,
#     intervals::Bool=false)  # returns spiketimes if false, intervals if true  


#     I = zeros(N)    # allocate vector for sample of size N 

#     dt = DEFAULT_SIMULATION_DT  # just to be clear

#     # extract Exwald parameters
#     (mu, lambda, tau) = Exwald_param

#     # First passage time model parameters for spontaneous Wald component 
#     (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

#     # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
#     trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

#     # drift rate  = spontaneous + stimulus
#     q(t) = v0 + stimulus(t)

#     #@infiltrate

#     # Exwald samples by simulating physical model of FPT + Poisson process in series
#     Exwald_simulate(I, q, s, barrier, trigger, dt)

#     # spike train is cumulative sum of intervals
#     return intervals ? I : cumsum(I)

# end

# # Simulate inhomogenous Exwald for T seconds
# # spontaneous Exwald parameters Exwald_param = (mu, lambda, tau)
# # stimulus function of time, default f(t)=0.0 (gives spontaneous spike train)
# # Default stimulus = 0.0 (spontaneous activity)
# function Exwald_Neuron(T::Float64,
#     Exwald_param::Tuple{Float64,Float64,Float64},
#     stimulus::Function,
#     dt::Float64=DEFAULT_SIMULATION_DT)  

#     # extract Exwald parameters
#     (mu, lambda, tau) = Exwald_param


#     # First passage time model parameters for spontaneous Wald component 
#     (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

#     # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
#     trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

#     # drift rate  = spontaneous + stimulus
#     q(t) = v0 + stimulus(t)

#     # allocate array for spike train, 2x longer than expected length
#     Expected_Nspikes = Int(round( T/(mu + tau))) # average number of spikes
#     spiketime = zeros(2*Expected_Nspikes)


#     # @infiltrate   # << this was active when I was last working on this ... 

#     # Exwald samples by simulating physical model of FPT + Poisson process in series
#     nSpikes = Exwald_simulate(spiketime, T, q, s, barrier, trigger, dt)

#     # spike train is cumulative sum of intervals
#     return spiketime[1:nSpikes]

# end

# returns neuron = (exwald_neuron, EXW_param) 
# where neuron is a closure function to simulate exwald_neuron(δ(t)) given cupula deflection δ(t)
# and EXW_param = (μ, λ, τ) is a tuple containing the neuron's parameters
# closure returns true if the neuron spiked in current time interval, false otherwise
function make_Exwald_neuron(EXW_param::Tuple{Float64, Float64, Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract Exwald parameters
    (mu, lambda, tau) = EXW_param


    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)
    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    G = 1.0  # mean input to exwald components is v = v0 + G*δ
    x = 0.0
    refractory = true

    function exwald_neuron(δ::Function, t::Float64)
        
        if refractory 
            dx = (v0 + G*δ(t))*dt + s * randn(1)[] * sqrt(dt)   # integrate noise
            x = x + dx
            if x >= barrier      
                refractory = false    # refractory period is over when barrier is reached
                x -= barrier     # reset integral for next refractory period
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



# returns neuron = (OU_neuron, OU_param) 
# where neuron is a closure function to simulate Ornstein_Uhlenbeck_Drift(δ(t)) 
# given cupula deflection δ(t), OU_param = (μ, λ, τ) is a tuple containing the neuron's parameters
# closure returns true if the neuron spiked in current time interval, false otherwise
function make_OU_neuron(OU_param::Tuple{Float64, Float64, Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract OU parameters
    (mu, lambda, tau) = OU_param


    # First passage time model parameters for τ = 0.0 (Inverse Gaussian/Wald model)
    # with barrier height = 1.0
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)

    G = 1.0  # mean input to exwald components is v = v0 + G*δ
    x = 0.0
        Random.seed!(4242)
    function ou_neuron(δ::Function, t::Float64)
        


        dx = (-x/tau + v0 + G*δ(t))*dt + s * randn(1)[] * sqrt(dt)   # leaky-integrate noise

        x = x + dx
        if x >= barrier      
            x -= barrier     # reset integral for next refractory period
            return true
        else
            return false  # still refractory
        end

    end

    return ou_neuron  #, OU_param
end


# returns SLIFneuron that takes 1 step in Ornstein-Uhlenbeck leaky drift-diffusion process
# with coloured noise input, given cupula deflection δ(t)
# closure returns true if v(t) reaches the barrier v==1 (neuron spiked), otherwise false
# assumes input gain g=1, to be revisited
function make_SLIF_neuron(SLIFparam::Tuple{Float64, Float64, Float64}, colour = 0.1, dt::Float64=DEFAULT_SIMULATION_DT)
# deprecated but not dead yet ...

    # extract SLIF parameters
    (v0, sigma, tau) = SLIFparam

    barrier = 1.0
    g = 1.0  
    v = 0.0

    # SDE coeffs pre-computed
    # A = dt/colour    
    # B = dt*sigma/sqrt(colour)

    # initial noise 0.0
    #z = 0.0

    pink = make_pink_noise(20)

    # set seed for debugging
    #Random.seed!(4242)

    function SLIFneuron(δ::Function, t::Float64) # deprecated but not dead yet ...

        # coloured noise  dZ = -(1/tau_n)*Z*dt + sigma/sqrt(tau_n)*dW
        #  where dW is Wiener process (Brownian motion) step
        # dz = -A*z + B*rand(Normal())
        # z = z + dz
        # z = z + dz

        d = 1.0   # coeff of v in C dv/vt + d*v = I
        # leaky integrate with coloured noise input
        # TBD parameter tau here should be C (such that tau = d/C)
        # dv = ( v0 + g*δ(t) - d*v/tau )*dt + sigma*pink()*sqrt(dt)
        dv = ( v0 + g*δ(t) - d*v )*dt/tau + sigma*pink()*sqrt(dt)

        v = v + dv
        if v >= barrier      
            v = 0.0 # -= barrier    # reset integral 
            return true     # spike
        else
            return false    # no spike
        end

    end

    return SLIFneuron 
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

# returns N interspike intervals generate by neuron responding to input(t) 
# exits & returns intervals so far if an interval exceeds timeout.
# (length(interval)<N if there was a timeout)
function interspike_intervals(neuron::Function, input::Function, N::Int64, 
                timeout::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    interval = zeros(Float64, N)

    t = 0.0
    previous_spiketime = 0.0
    i = 0
    while i < N
        t += dt
        Delta_t = t - previous_spiketime
        if Delta_t > timeout
            return interval[1:i]
        elseif neuron(input, t)==true   # neuron fired
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

# returns closure for fractional torsion pendulum state update 
#  given input u(t) = head angular velocity or acceleration.
#  TP_update(u::Function, t::Float64)  # closure
#  If v==true (default) then u(t) is head angular velocity, else it's acceleration.
function make_fractional_torsion_pendulum_stateUpdate(
    q::Float64, y0::Float64, yp0::Float64, v::Bool=true, 
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
    function update(u::Function, t::Float64)

        if v==true      # velocity input specified
            wdot = diffcd(u, t, dt)  # differentiate input
        else
            wdot = u(t)  
        end

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


"""
    fpt_ou_pdf(t, x0, kappa, mu, sigma, b; N=2000)

Compute the probability density function of the first passage time for an
Ornstein-Uhlenbeck process dX_t = kappa * (mu - X_t) * dt + sigma * dW_t,
starting at X_0 = x0, to hit the barrier b.

This uses the semi-analytical method of Lipton & Kaushansky (2018) involving
solution of a Volterra integral equation via discretization. Handles both upper
and lower barriers via reflection symmetry.

# Arguments
- `t::Real`: Time at which to evaluate the density (>0).
- `x0::Real`: Starting value.
- `kappa::Real >0`: Reversion speed.
- `mu::Real`: Long-term mean.
- `sigma::Real >0`: Volatility.
- `b::Real`: Barrier level.
- `N::Int=2000`: Number of grid points for discretization (higher for accuracy).

# Returns
- `Float64`: The FPT density g(t) = d/dt P(τ_b ≤ t).

Note: For barrier at mu (special case), a closed-form exists but is not used here.
"""
function fpt_ou_pdf(t::Real, x0::Real, kappa::Real, mu::Real, sigma::Real, b::Real; N::Int=2000)
    if t <= 0
        return 0.0
    end
    if kappa <= 0 || sigma <= 0
        error("kappa and sigma must be positive")
    end

    alpha = sqrt(kappa) / sigma
    y0 = alpha * (x0 - mu)
    c = alpha * (b - mu)

    t_bar = kappa * t
    if y0 > c
        g_bar = ou_fpt_density_std(t_bar, y0, c, N=N)
    else
        g_bar = ou_fpt_density_std(t_bar, -y0, -c, N=N)
    end
    return kappa * g_bar
end

"""
    ou_fpt_density_std(t, z, b; N=2000)

Standardized version: density for dX_t = -X_t dt + dW_t, first hitting time to b
from z > b (handles b of either sign via the general Volterra formulation).
"""
function ou_fpt_density_std(t::Real, z::Real, b::Real; N::Int=2000)
    if t <= 0
        return 0.0
    end
    if b >= z
        error("In standardized case, z > b required")
    end

    et = exp(t)
    e2t = et * et
    denom = e2t - 1.0
    tau = denom / 2.0

    # Term 1: Free (image) term
    diffbz = et * b - z
    exp_arg = - (diffbz^2) / denom + 2.0 * t
    term1 = - diffbz * exp(exp_arg) / sqrt(pi * denom^3)

    # Discretize [0, tau] with N intervals, N+1 points
    h = tau / N
    taus = [k * h for k in 0:N]
    nu = zeros(N + 1)
    nu[1] = 0.0  # At tau=0

    # Solve Volterra: nu(tau) = -free(tau) - ∫_0^tau K(tau, s) nu(s) ds
    # Using left rectangle rule for integral
    for i in 2:(N + 1)  # i=1 is 0
        ti = taus[i]
        bt_i = b * sqrt(2 * ti + 1.0)
        free_i = exp( - (bt_i - z)^2 / (2 * ti) ) / sqrt(2 * pi * ti )

        integ = 0.0
        for j in 1:(i - 1)
            tj = taus[j]
            bt_j = b * sqrt(2 * tj + 1.0)
            delta_bt = bt_i - bt_j
            delta_t = ti - tj
            exp_k = exp( - delta_bt^2 / (2 * delta_t) )
            k_j = delta_bt * exp_k / sqrt(2 * pi * delta_t^3)
            integ += h * k_j * nu[j]
        end
        nu[i] = - free_i - integ
    end

    nu_tau = nu[end]

    # Term 2: Local term with nu(tau)
    term2 = - (et * b + e2t / sqrt(pi * denom)) * nu_tau

    # Term 3: Integral term
    term3 = 0.0
    coef3 = e2t / sqrt(8 * pi)
    for j in 1:N  # j=1 to N, taus[N+1]=tau, but nu[N+1]=nu_tau
        tj = taus[j]
        delta_t = tau - tj
        if delta_t <= 0
            continue
        end
        nu_diff = nu[j] - nu_tau
        bracket = 1.0 - 4.0 * b^2 * delta_t
        exp_part = exp(-2.0 * b^2 * delta_t)
        integrand_j = bracket * exp_part * nu_diff / (delta_t ^ 1.5)
        term3 += h * integrand_j
    end
    term3 *= coef3

    return term1 + term2 + term3
end

"""
    fpt_ig_leaky_pdf(t, mu_ig, lambda, tau; N=2000)

Compute the FPT density for a leaky integrate-and-fire model (OU process), parameterized
by the inverse Gaussian parameters (mu_ig, lambda) for the non-leaky case (tau = ∞)
and the reversion time constant tau > 0.

- Starts at x0 = 0, hits upper barrier b = 1.
- When tau → ∞ (kappa → 0), recovers IG(mu_ig, lambda) density.
- kappa = 1 / tau (reversion speed).
- sigma = 1 / sqrt(lambda) (volatility).
- mu_ou = tau / mu_ig (long-term mean, scaled to match IG drift nu = 1 / mu_ig).

# Arguments
- `t::Real`: Time (>0).
- `mu_ig::Real >0`: Mean of the limiting IG distribution.
- `lambda::Real >0`: Shape of the limiting IG distribution.
- `tau::Real >0`: Reversion time constant.
- `N::Int=2000`: Discretization points.

# Returns
- `Float64`: FPT density at t.
"""
function make_fpt_ig_leaky_pdf(mu_ig::Real, lambda::Real, tau::Real)

    # if t <= 0
    #     return 0.0
    # end
    if mu_ig <= 0 || lambda <= 0 || tau <= 0
        error("mu_ig, lambda, tau must be positive")
    end

    kappa = 1.0 / tau
    sigma = 1.0 / sqrt(lambda)
    x0 = 0.0
    b = 1.0
    mu_ou = tau / mu_ig

    # closure
    function OU_FPT_pdf(t::Real, N::Int64 = 2000)
        return fpt_ou_pdf(t, x0, kappa, mu_ou, sigma, b; N=N)
    end

    return OU_FPT_pdf

end

# Optional: Limiting IG density for verification (tau = ∞)
"""
    ig_pdf(t, mu_ig, lambda)

Inverse Gaussian density: limiting case as tau → ∞.
"""
function ig_pdf(t::Real, mu_ig::Real, lambda::Real)
    if t <= 0
        return 0.0
    end
    return sqrt(lambda / (2 * pi * t^3)) * exp( -lambda * (t - mu_ig)^2 / (2 * mu_ig^2 * t) )
end


# transform SLIF parameters (μₛ, λₛ, τₛ) to Exwald parameters (μₓ, λₓ, τₓ)
# By trilinear interpolation in 3d grid given by
# EXWparam, (mu_o, lambda_o,tau_o) computed by map_OU2Exwald()
function SLIF2Exwald(OUparam::Tuple{Float64, Float64, Float64},
                    grid::Tuple{Vector{Float64},Vector{Float64}, Vector{Float64}},
                    O2X_map::Array{Tuple{Float64, Float64, Float64}, 3})
    # log transform
    mu_o, lambda_o, tau_o = log.(OUparam)
    muo_grid = log.(grid[1])
    lambdao_grid = log.(grid[2])
    tauo_grid = log.(grid[3])

    n = length(muo_grid)
    if n != size(O2X_map, 1) || n != size(O2X_map, 2) || n != size(O2X_map, 3)
        throw(ArgumentError("Grids must have length equal to O2X dimensions (40)"))
    end

    # Find voxel indices (1-based)
    i = searchsortedfirst(muo_grid, mu_o) - 1
    j = searchsortedfirst(lambdao_grid, lambda_o) - 1
    k = searchsortedfirst(tauo_grid, tau_o) - 1

    if i < 1 || i >= n || muo_grid[i] > mu_o || muo_grid[i+1] < mu_o ||
       j < 1 || j >= n || lambdao_grid[j] > lambda_o || lambdao_grid[j+1] < lambda_o ||
       k < 1 || k >= n || tauo_grid[k] > tau_o || tauo_grid[k+1] < tau_o
        return (NaN, NaN, NaN)
    end

    # Check all 8 vertices valid
    valid = true
    for ii in 0:1, jj in 0:1, kk in 0:1
        if isnan(log(O2X_map[i + ii, j + jj, k + kk][1]))
            valid = false
            break
        end
    end
    if !valid
        return (NaN, NaN, NaN)
    end


    # Normalized coordinates
    xd = (mu_o - muo_grid[i]) / (muo_grid[i + 1] - muo_grid[i])
    yd = (lambda_o - lambdao_grid[j]) / (lambdao_grid[j + 1] - lambdao_grid[j])
    zd = (tau_o - tauo_grid[k]) / (tauo_grid[k + 1] - tauo_grid[k])

    # Build vals for each component
    mu_vals = zeros(2, 2, 2)
    lambda_vals = zeros(2, 2, 2)
    tau_vals = zeros(2, 2, 2)
    for ii in 0:1, jj in 0:1, kk in 0:1
        t = log.(O2X_map[i + ii, j + jj, k + kk])
        mu_vals[ii + 1, jj + 1, kk + 1] = t[1]
        lambda_vals[ii + 1, jj + 1, kk + 1] = t[2]
        tau_vals[ii + 1, jj + 1, kk + 1] = t[3]
    end

    mux = trilinear_normalized(xd, yd, zd, mu_vals)
    lambdax = trilinear_normalized(xd, yd, zd, lambda_vals)
    taux = trilinear_normalized(xd, yd, zd, tau_vals)

    return exp.((mux, lambdax, taux))
end

# transform Exwald parameters (μₓ, λₓ, τₓ) to  SLIF parameters (μₛ, λₛ, τₛ)
# By Newton-Raphson search in 3d grid given by
# O2X_map, grid = (mu_o, lambda_o,tau_o) computed by map_OU2Exwald()
function Exwald2SLIF(EXWparam::Tuple{Float64, Float64, Float64},
                    grid::Tuple{Vector{Float64}, Vector{Float64},Vector{Float64}},
                    O2X_map::Array{Tuple{Float64, Float64, Float64}, 3})
    
    # log transform
    mu_x, lambda_x, tau_x = log.(EXWparam)
    muo_grid     = log.(grid[1])
    lambdao_grid = log.(grid[2])
    tauo_grid    = log.(grid[3])
   # O2X_map = log.(O2X_map)

    n = length(muo_grid)
    if n != size(O2X_map, 1) || n != size(O2X_map, 2) || n != size(O2X_map, 3)
        throw(ArgumentError("Grids must have length equal to O2X dimensions (40)"))
    end

    # Compute global bounds from valid points
    mu_xs = Float64[]
    lambda_xs = Float64[]
    tau_xs = Float64[]
    for i in 1:n, j in 1:n, k in 1:n
        t = log.(O2X_map[i, j, k])
        if !isnan(t[1])
            push!(mu_xs, t[1])
            push!(lambda_xs, t[2])
            push!(tau_xs, t[3])
        end
    end
    if isempty(mu_xs)
        return (NaN, NaN, NaN)
    end
    min_mu_x, max_mu_x = minimum(mu_xs), maximum(mu_xs)
    min_lambda_x, max_lambda_x = minimum(lambda_xs), maximum(lambda_xs)
    min_tau_x, max_tau_x = minimum(tau_xs), maximum(tau_xs)

    if !(min_mu_x <= mu_x <= max_mu_x &&
         min_lambda_x <= lambda_x <= max_lambda_x &&
         min_tau_x <= tau_x <= max_tau_x)
        return (NaN, NaN, NaN)
    end

    # Loop over voxels
    for i in 1:(n - 1), j in 1:(n - 1), k in 1:(n - 1)
        # Check valid
        valid = true
        for ii in 0:1, jj in 0:1, kk in 0:1
            if isnan(log.(O2X_map[i + ii, j + jj, k + kk][1]))
                valid = false
                break
            end
        end
        if !valid
            continue
        end

        # Build vals
        mu_vals = zeros(2, 2, 2)
        lambda_vals = zeros(2, 2, 2)
        tau_vals = zeros(2, 2, 2)
        for ii in 0:1, jj in 0:1, kk in 0:1
            t = log.(O2X_map[i + ii, j + jj, k + kk])
            mu_vals[ii + 1, jj + 1, kk + 1] = t[1]
            lambda_vals[ii + 1, jj + 1, kk + 1] = t[2]
            tau_vals[ii + 1, jj + 1, kk + 1] = t[3]
        end

        # Quick AABB check
        if !(minimum(mu_vals) <= mu_x <= maximum(mu_vals) &&
             minimum(lambda_vals) <= lambda_x <= maximum(lambda_vals) &&
             minimum(tau_vals) <= tau_x <= maximum(tau_vals))
            continue
        end

        # Newton-Raphson
        params = [0.5, 0.5, 0.5]
        tol = 1e-10
        maxiter = 1000
        for _ in 1:maxiter
            xd, yd, zd = params

            # Residual
            rm = trilinear_normalized(xd, yd, zd, mu_vals) - mu_x
            rl = trilinear_normalized(xd, yd, zd, lambda_vals) - lambda_x
            rt = trilinear_normalized(xd, yd, zd, tau_vals) - tau_x
            r = [rm, rl, rt]
            res_norm = norm(r)
            if res_norm < tol
                # Check bounds
                if all(0 .<= params .<= 1)
                    # Compute pre-image
                    mu_o = muo_grid[i] * (1 - params[1]) + muo_grid[i + 1] * params[1]
                    lambda_o = lambdao_grid[j] * (1 - params[2]) + lambdao_grid[j + 1] * params[2]
                    tau_o = tauo_grid[k] * (1 - params[3]) + tauo_grid[k + 1] * params[3]
                    return exp(mu_o), exp(lambda_o), exp(tau_o)
                end
                break
            end

            # Jacobian
            J = zeros(3, 3)

            # mu row
            diff_yz_mu = zeros(2, 2)
            diff_yz_mu[1, 1] = mu_vals[2, 1, 1] - mu_vals[1, 1, 1]
            diff_yz_mu[1, 2] = mu_vals[2, 1, 2] - mu_vals[1, 1, 2]
            diff_yz_mu[2, 1] = mu_vals[2, 2, 1] - mu_vals[1, 2, 1]
            diff_yz_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[1, 2, 2]
            J[1, 1] = bilinear_normalized(yd, zd, diff_yz_mu)

            diff_xz_mu = zeros(2, 2)
            diff_xz_mu[1, 1] = mu_vals[1, 2, 1] - mu_vals[1, 1, 1]
            diff_xz_mu[2, 1] = mu_vals[2, 2, 1] - mu_vals[2, 1, 1]
            diff_xz_mu[1, 2] = mu_vals[1, 2, 2] - mu_vals[1, 1, 2]
            diff_xz_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[2, 1, 2]
            J[1, 2] = bilinear_normalized(xd, zd, diff_xz_mu)

            diff_xy_mu = zeros(2, 2)
            diff_xy_mu[1, 1] = mu_vals[1, 1, 2] - mu_vals[1, 1, 1]
            diff_xy_mu[2, 1] = mu_vals[2, 1, 2] - mu_vals[2, 1, 1]
            diff_xy_mu[1, 2] = mu_vals[1, 2, 2] - mu_vals[1, 2, 1]
            diff_xy_mu[2, 2] = mu_vals[2, 2, 2] - mu_vals[2, 2, 1]
            J[1, 3] = bilinear_normalized(xd, yd, diff_xy_mu)

            # lambda row
            diff_yz_lambda = zeros(2, 2)
            diff_yz_lambda[1, 1] = lambda_vals[2, 1, 1] - lambda_vals[1, 1, 1]
            diff_yz_lambda[1, 2] = lambda_vals[2, 1, 2] - lambda_vals[1, 1, 2]
            diff_yz_lambda[2, 1] = lambda_vals[2, 2, 1] - lambda_vals[1, 2, 1]
            diff_yz_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[1, 2, 2]
            J[2, 1] = bilinear_normalized(yd, zd, diff_yz_lambda)

            diff_xz_lambda = zeros(2, 2)
            diff_xz_lambda[1, 1] = lambda_vals[1, 2, 1] - lambda_vals[1, 1, 1]
            diff_xz_lambda[2, 1] = lambda_vals[2, 2, 1] - lambda_vals[2, 1, 1]
            diff_xz_lambda[1, 2] = lambda_vals[1, 2, 2] - lambda_vals[1, 1, 2]
            diff_xz_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[2, 1, 2]
            J[2, 2] = bilinear_normalized(xd, zd, diff_xz_lambda)

            diff_xy_lambda = zeros(2, 2)
            diff_xy_lambda[1, 1] = lambda_vals[1, 1, 2] - lambda_vals[1, 1, 1]
            diff_xy_lambda[2, 1] = lambda_vals[2, 1, 2] - lambda_vals[2, 1, 1]
            diff_xy_lambda[1, 2] = lambda_vals[1, 2, 2] - lambda_vals[1, 2, 1]
            diff_xy_lambda[2, 2] = lambda_vals[2, 2, 2] - lambda_vals[2, 2, 1]
            J[2, 3] = bilinear_normalized(xd, yd, diff_xy_lambda)

            # tau row
            diff_yz_tau = zeros(2, 2)
            diff_yz_tau[1, 1] = tau_vals[2, 1, 1] - tau_vals[1, 1, 1]
            diff_yz_tau[1, 2] = tau_vals[2, 1, 2] - tau_vals[1, 1, 2]
            diff_yz_tau[2, 1] = tau_vals[2, 2, 1] - tau_vals[1, 2, 1]
            diff_yz_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[1, 2, 2]
            J[3, 1] = bilinear_normalized(yd, zd, diff_yz_tau)

            diff_xz_tau = zeros(2, 2)
            diff_xz_tau[1, 1] = tau_vals[1, 2, 1] - tau_vals[1, 1, 1]
            diff_xz_tau[2, 1] = tau_vals[2, 2, 1] - tau_vals[2, 1, 1]
            diff_xz_tau[1, 2] = tau_vals[1, 2, 2] - tau_vals[1, 1, 2]
            diff_xz_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[2, 1, 2]
            J[3, 2] = bilinear_normalized(xd, zd, diff_xz_tau)

            diff_xy_tau = zeros(2, 2)
            diff_xy_tau[1, 1] = tau_vals[1, 1, 2] - tau_vals[1, 1, 1]
            diff_xy_tau[2, 1] = tau_vals[2, 1, 2] - tau_vals[2, 1, 1]
            diff_xy_tau[1, 2] = tau_vals[1, 2, 2] - tau_vals[1, 2, 1]
            diff_xy_tau[2, 2] = tau_vals[2, 2, 2] - tau_vals[2, 2, 1]
            J[3, 3] = bilinear_normalized(xd, yd, diff_xy_tau)

            # Update
            delta = J \ (-r)
            params .+= delta
            params = clamp.(params, 0.0, 1.0)
        end

        # Final check
        xd, yd, zd = params
        rm = trilinear_normalized(xd, yd, zd, mu_vals) - mu_x
        rl = trilinear_normalized(xd, yd, zd, lambda_vals) - lambda_x
        rt = trilinear_normalized(xd, yd, zd, tau_vals) - tau_x
        final_res = norm([rm, rl, rt])
        if final_res < tol && all(0 .<= params .<= 1)
            mu_o = muo_grid[i] * (1 - xd) + muo_grid[i + 1] * xd
            lambda_o = lambdao_grid[j] * (1 - yd) + lambdao_grid[j + 1] * yd
            tau_o = tauo_grid[k] * (1 - zd) + tauo_grid[k + 1] * zd
            return exp(mu_o), exp(lambda_o), exp(tau_o)
        end
    end

    return (NaN, NaN, NaN)
end

# transform Exwald parameters (μₓ, λₓ, τₓ) to  SLIF parameters (μₛ, λₛ, τₛ)
# By Newton-Raphson search in 3d grid given by
# O2X_map, grid = (mu_o, lambda_o,tau_o) computed by map_OU2Exwald()
# stored in jld2file 
function Exwald2SLIF(EXWparam::Tuple{Float64, Float64, Float64},
                     JLD2_filename::String = "OU2EXW_40x40x40_4000_1.jld2")

    # load OU2EXW map
    DATA=load(JLD2_filename);
    O2X_map = DATA["EXWparam"]
    grid = DATA["vex"]                

    OUparam = Exwald2SLIF(EXWparam,grid, O2X_map)

end

# returns a function that returns Exwald parameters given SLIF parameters
function dev1_parameterize_OU2EXW(i::Int64, j::Int64, a::Float64)

    # load OU2EXW map
    DATA=load("OU2EXW_40x40x40_4000_1.jld2");
    EXWparam = DATA["EXWparam"]
    grid = DATA["vex"]

    # 
    mu_grid = grid[1]
    lambda_grid = grid[2]
    tau_grid = grid[3]
    Nt = length(tau_grid)

    # data (contains NaNs)
    tau = [EXWparam[i, j, k][3] for k in 1:Nt]

    # extract segment containing Reals
    k0 = findfirst(!isnan, tau)
    k1 = findlast(!isnan, tau)
    K = k0:k1

    # initial estimates
    A0 = 1.0*1.2
    B0 = 0.145*tau[k0]*1.2
    C0 = 0.67*tau[k1]*1.2
    pInit = [A0, B0, C0]
    grad = zeros(length(pInit))  # required but not used in optimization

    # bounds
    LB = [0.0, 0.0, 0.0]
    UB = [Inf, Inf, Inf]

    # model
    f(x,p) =  exp(p[1]./(log.(x./p[2])) .+ log(p[3]))

    # error functional (sum squared error model - data)
    # returns Inf for invalid parameters
    function V(param, grad) 

        v = 0.0
        for k in K
            try 
                v += (f(tau_grid[k], param)-tau[k])^2
            catch
                v += Inf
            end
        end

        return v 
    end

    # y = exp.(f(tau_grid, pInit))
    # replace!(e -> e <= C0 || e > 100.0 ? NaN : e, y)  # hide values < 0.0 in plot

    Prob = Optimization.OptimizationProblem(OptimizationFunction(V), pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_PRAXIS(), reltol = 1.0e-9)
 
    println(sol.u)
    # fitted curve
    y = [f(tau_grid[k], sol.u) for k in K ]

    # init curve
    yx = [f(tau_grid[k], pInit) for k in K ]

    println("μ = ", mu_grid[i], ", λ = ", lambda_grid[j])

    FF = Figure()
    ax = Axis(FF[1,1], xscale = log10, yscale = log10,
         xtickformat = "{:.4f}", ytickformat = "{:.5f}")

    scatter!(tau_grid[K], tau[K], color = :skyblue, markersize = 12)


    lines!(tau_grid[K], y, color = :salmon, linewidth = 3)

  #  lines!(tau_grid[K], yx, color = :blue)

    display(FF)

end


# returns a function that returns Exwald parameters given SLIF parameters
function dev2_parameterize_OU2EXW()

    # load OU2EXW map
    DATA=load("OU2EXW_40x40x40_4000_1.jld2");
    EXWparam = DATA["EXWparam"]
    grid = DATA["vex"]



    # 
    mu_grid = grid[1]
    Nmu = length(mu_grid)
    lambda_grid = grid[2]
    Nlam = length(lambda_grid)
    tau_grid = grid[3]
    Ntau = length(tau_grid)

       # model
    f(x,p) =  exp(p[1]./(log.(x./p[2])) .+ log(p[3]))

    # error functional (sum squared error model - data)
    # returns Inf for invalid parameters

    grad = zeros(3)  # required but not used in optimization

    # bounds
    LB = [0.0, 0.0, 0.0]
    UB = [Inf, Inf, Inf]

    fittedparam = NaN*zeros(Nmu, Nlam, 3)

    for i in 1:(Nmu-3)

    FF = Figure()
    ax = Axis(FF[1,1], xscale = log10, yscale = log10,
         xtickformat = "{:.4f}", ytickformat = "{:.5f}")

    for j in 1:Nlam

        # data (contains NaNs)
        tau = [EXWparam[i, j, k][3] for k in 1:Ntau]

        # extract segment containing Reals
        k0 = findfirst(!isnan, tau)
        k1 = findlast(!isnan, tau)
        K = k0:k1

        function V(param, grad) 

            v = 0.0
            for k in K
                try 
                    v += (log(f(tau_grid[k], param))-log(tau[k]))^2
                catch
                    v += Inf
                end
            end

            return v 
        end


        # initial estimates
        # A0 = 1.0
        # B0 = 0.145*tau[k0]
        # C0 = 0.67*tau[k1]
        pInit = [2.0, 0.001*tau[k0], tau[k1]]
 
        # if i == 1
        #     pInit = [2.0, 0.001*tau[k0], tau[k1]]
        # else
        #     pInit = fittedparam[i-1,:]
        # end

        Prob = Optimization.OptimizationProblem(OptimizationFunction(V), pInit, grad, lb=LB, ub = UB)
        sol = solve(Prob, NLopt.LN_PRAXIS(), reltol = 1.0e-9)

        fittedparam[i,j, :] = sol.u
    
       # println(sol.u)
        # fitted curve
        y = [f(tau_grid[k], sol.u) for k in K ]

        # init curve
        yx = [f(tau_grid[k], pInit) for k in K ]

    #  println("μ = ", mu_grid[i], ", λ = ", lambda_grid[j])

        scatter!(ax, tau_grid[K], tau[K], color = :maroon, markersize = 6)

        lines!(ax, tau_grid[K], y, color = :salmon, linewidth = 1)

    end

  #  lines!(tau_grid[K], yx, color = :blue)

    println(i)
    display(FF)

end

    return fittedparam

end

function make_fractional_SLIF_neuron(
    SLIF_param::Tuple{Float64, Float64, Float64}, q::Float64, 
    x0::Float64=0.0; 
    dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

    # # Fractional Steinhausen model: I.y'' + P.y' + K.Dq y = I.wdot, 
    # I = 2.0e-12   # coeff of q'', endolymph moment of inertia kg.m^2
    # P = 6.0e-11   # coeff of q', viscous damping N.m.s/rad 
    # G = 1.0e-10    # coeff of q, cupula stiffness N.m/rad 

    # # Update equation coeffs from model parameters
    # A = P/I
    # B = G/I

    # Fractional SLIF neuron
        # extract OU parameters
    (mu, lambda, tau) = SLIF_param

    # First passage time model parameters for τ = 0.0 (Inverse Gaussian/Wald model)
    # with barrier height = 1.0
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)
 

    # input gain (how much the drift rate is affected by input)
    G = 1.0

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

    # Augmented state: x = [y, v1, ..., vM]
    x = [x0; zeros(M)]
    du = zeros(length(x))

    Threshold = 1.0
    Random.seed!(4242)

    # neuron update function given u(t) 
    function qSLIF(u::Function, t::Float64)

     #   @infiltrate

     #   dx = (-x/tau + v0 + G*u(t))*dt + s * randn(1)[] * sqrt(dt) 
 

        ut = v0 + G*u(t) + s*sqrt(tau*mu^3/lambda)*randn(1)[]/sqrt(dt)   # input at t

        vs = @view x[2:end]

       # @infiltrate
        
        # Approximate D^q u 
        if (q==0.0) 
            approx_dq = ut
        else
            approx_dq = K * ut
            for i in 1:M
                approx_dq += residues[i] * vs[i]
            end
        end
        
        # state update
        du[1] =  approx_dq - x[1]/tau
        
        # Auxiliary state update
        for i in 1:M
            du[1 + i] = -p_i[i] * vs[i] + ut
        end

     #   @infiltrate

        # Euler integration
        for i in 1:length(x)
            x[i] += du[i] * dt
        end

        if x[1] >= Threshold      
            x[1] -= Threshold 
           # vs .= 0.0
            return true
        else
            return false  
        end
    
    end

    # return closure
    return qSLIF 
end
