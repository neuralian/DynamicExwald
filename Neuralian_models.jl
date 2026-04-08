# Neuralian Toolbox
# Tools for simulating spiking neurons and canal dynamics 
# MGP 2024-26

# construct Poisson neuron (Threshold-crossing, Gaussian noise with time-varying mean)
# Poisson_neuron(u(t), t) returns true if neuron fired on current timestep given input u(t)
# Poisson neuron fires if noise crosses threshold, input u(t) changes the noise mean.
# refractory period prevents very short intervals, specified in units of dt
# When u(t) = t -> 0.0, interspike intervals have Exponential(tau) distribution 
# nb returns Tuple, the neuron function is 1st element
function make_Poisson_neuron(tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    trigger = Poisson_Trigger_Threshold(tau)
    G = 1.0  # gain u(t)-> change in noise mean 

    function Poisson_neuron(u::Function, t::Float64)

        if (G*u(t)+randn()[]) >= trigger   
            return true   # fired
        else
            return false  # didn't fire
        end

    end

    return Poisson_neuron, tau 
end

# construct Wald neuron (drift-diffusion to barrier a.k.a. stochastic integrate and fire)
# Wald_neuron(u(t), t) returns true if neuron fired on current timestep given input u(t)
# Wald neuron integrates Gaussian noise to threshold, fires and resets.
# When u(t) = t -> 0.0, interspike intervals have a Wald a.k.a. Inverse Gaussian distribution IG(μ,λ). 
# nb/warning returns tuple containing wald_neuron() and drift-diffusion coeffs
function make_Wald_neuron(mu::Float64, lambda::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # First passage time model parameters for τ = 0.0 (Inverse Gaussian/Wald model)
    # with barrier height = 1.0
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)

    G = 1.0  # mean input to exwald components is v = v0 + G*δ
    x = Ref(0.0)

    function wald_neuron(u::Function, t::Float64)

        # drift  +  diffusion
        dx = ( v0 + G*u(t) )*dt  +  s*randn(1)[]*sqrt(dt)   

        x[] = x[] + dx
        if x[] >= barrier      
            x[] -= barrier     # reset integral for next refractory period
            return true
        else
            return false  # still refractory
        end

    end

    return wald_neuron, (v0, s, barrier) 
end

# Returns SLIF_neuron() that simulates 1 timestep (dt) of SLIF neuron
# SLIF_neuron returns true if the neuron fired, false otherwise.
# Also returns the parameters (a, σ_v, tau) of the SLIF model
#  dV   = (a + u(t) - V[] / tau) * dt + σ_v * sqrt_dt * randn()
# Input Params (mu, lambda) are parameters of Inverse Gaussian ISI distribution in the limit of large tau
#              tau is SLIF neuron membrane time constant (distinct from Exwald tau)
# 25MARCH26
function make_SLIF_neuron(
    mu::Float64,
    lambda::Float64,
    tau::Float64;
    dt::Float64 = DEFAULT_SIMULATION_DT
)
    V_reset = 0.0
    V_th    = 1.0
    θ       = V_th - V_reset

    a       = θ / mu
    σ_v     = 1.0 / sqrt(lambda)

    C       = 1e-9
    g       = C / tau
    s       = C * σ_v

    V       = Ref(V_reset)
    sqrt_dt = sqrt(dt)

    function SLIF_neuron(u::Function, t::Float64)::Bool
        dV   = (a + u(t) - V[] / tau) * dt + σ_v * sqrt_dt * randn()
        V[] += dV
        if V[] >= V_th
            V[] = V[] - V_th
            return true
        end
        return false
    end

    return SLIF_neuron, (a, σ_v, tau )
end

# qSLIF neuron constructor using qSLIF model coeffs
# Default fractional order integration bandwidth 0.01-20Hz
# 25MARCH26
function make_qSLIFc_neuron(a::Float64, sigma::Float64, tau_memb::Float64, q::Float64=1.0; 
    dt::Float64         = DEFAULT_SIMULATION_DT,
    qOrder::Int              = 5,
    omega_low::Float64  = 2π * 0.01,
    omega_high::Float64 = 2π * 20.0
)

    # ── LIF parameters (same as before) ─────────────────────────────────────
    V_reset = 0.0
    V_th    = 1.0
    θ       = V_th - V_reset
    # a       = θ / mu
    # σ_v     = 1.0 / sqrt(lambda)
    # C       = 1e-9
    # g       = C / tau
    # s       = C * σ_v
    sqrt_dt = sqrt(dt)

    dq = make_fractional_derivative(q)

    # ── Mutable closure state ────────────────────────────────────────────────
    V  = Ref(V_reset)
    z  = zeros(Float64, qOrder)      # Oustaloup filter states

    # returns true/false if neuron fired/didnt, and membrane potential at current time step
    function qSLIF_neuron(u::Function, t::Float64)::Tuple{Bool, Float64}

        # Raw input: deterministic drive + noise (input-referred)
        x = u(t) + sigma * randn() / sqrt_dt   # [norm-V/s]; noise is input-referred

        # fractional differintegrator
        y = dq(x)
        
        # LIF membrane update — leak is intrinsic, fractional input replaces
        # the direct (a + u(t)) drive of the standard model
        dV   = (a + y - V[] / tau_memb) * dt
        V[] += dV

        fired = ( V[] >= V_th )
        if fired
            V[] = V[] - V_th
        end
        
        return fired, V[]
    end

    return qSLIF_neuron, (a,sigma, tau_memb, q)
end

# Returns closure qSLIF_neuron that simulates 1 timestep (dt) of fractional SLIF neuron
# qSLIF_neuron returns true if the neuron fired, false otherwise, plus the membrane potential V[]
# Also returns the parameters (a, σ_v, tau, q) of the fractional SLIF model
# NB recovers SLIF_neuron for q=1
# Input Param::Tuple is (mu, lambda, tau) where (mu, lambda) are parameters of 
#                       Inverse Gaussian ISI distribution in the limit of large tau
#                       and tau is the SLIF neuron membrane time constant (distinct from Exwald tau).
# Default fractional order integration bandwidth 0.01-20Hz
# 25MARCH26
function make_qSLIF_neuron(mu::Float64, lambda::Float64, tau::Float64, q::Float64=1.0; 
    dt::Float64         = DEFAULT_SIMULATION_DT,
    qOrder::Int              = 5,
    omega_low::Float64  = 2π * 0.01,
    omega_high::Float64 = 2π * 20.0
)

    # ── LIF parameters (same as before) ─────────────────────────────────────
    V_reset = 0.0
    V_th    = 1.0
    θ       = V_th - V_reset
    a       = θ / mu
    σ_v     = 1.0 / sqrt(lambda)
    C       = 1e-9
    g       = C / tau
    s       = C * σ_v
    sqrt_dt = sqrt(dt)

    dq = make_fractional_derivative(q)

    # ── Mutable closure state ────────────────────────────────────────────────
    V  = Ref(V_reset)
    z  = zeros(Float64, qOrder)      # Oustaloup filter states

    # returns true/false if neuron fired/didnt fire, and membrane potential V[]
    function qSLIF_neuron(u::Function, t::Float64)::Tuple{Bool, Float64}

        # Raw input: deterministic drive + noise (input-referred)
        x = u(t) + σ_v * randn() / sqrt_dt   # [norm-V/s]; noise is input-referred

        # fractional differintegrator
        y = dq(x)
        
        # LIF membrane update — leak is intrinsic, fractional input replaces
        # the direct (a + u(t)) drive of the standard model
        dV   = (a + y - V[] / tau) * dt
        V[] += dV

        if V[] >= V_th
            V[] = V[] - V_th
            return true
        end
        return false
    end

    return qSLIF_neuron, (a, σ_v, tau, q )
end

# qSLIF/Exwald example neurons 
# (hand-)fitted along PC1 of Paulin, Pullar & Hoffman (2024) Fig. 3D.
# i = 1: very regular, CV = 0.025
#     2: regular,      CV = 0.04
#     3: intermediate  CV = 0.1
#     4: irregular     CV = 0.27
#     5: v. irregular  CV = 0.57 
function qSLIF_example_neuron(i::Int64)

    # 5 parameter sets
    param = ( 
        (80.0,      0.17678,    0.025,   0.0),
        (142.85714, 0.1768,     0.01,    0.1),
        (200.0,     0.1768,     0.00525, 0.15),
        (142.85714, 0.1768,     0.006,   0.35),
        (125.0,     0.1768,     0.005,   0.4)
            )

    # Mean, SD, CV, CV*
    summarystats = (
        (0.01732, 0.00043, 0.02466, 0.02329),
        (0.01201, 0.00046, 0.03852, 0.04168),
        (0.01504, 0.00159, 0.10582, 0.10787),
        (0.02148, 0.00573, 0.26681, 0.19896),
        (0.04316, 0.02444, 0.56641, 0.27590)
    )

    neuron, _ = make_qSLIFc_neuron(param[i]...)

    return neuron, param, summarystats
end


# simulate neuron responding to u(t) up to time T
# neuron constructed by e.g. neuron, _ = make_Wald_neuron() 
# output is vector of spike times 
# exits with warning if an interval exceeds timeout (default 1s)
function spiketimes(neuron::Function, u::Function, T::Float64, 
        timeout::Float64 = 1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    # initialize spike time vector
    spt = Float64[]

    # time of previous spike
    t_prev = 0.0

    for t in 0.0:dt:T
        if neuron(u, t)[1]==true   # neuron fired
            push!(spt, t)
            t_prev = t
        elseif t-t_prev > timeout
            @printf "spiketimes() timed out"
            return spt
        end
    end

    return spt
end

#  N interspike intervals generated by neuron responding to stimulus u(t) 
# neuron constructed by e.g. neuron, _ = make_Wald_neuron()
# exits & returns intervals so far if an interval exceeds timeout.
function interspike_intervals(neuron::Function, u::Function, N::Int64, 
                timeout::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    ISI = zeros(Float64, N)
    t = 0.0
    t_prev = 0.0
    i = 0

    while i < N
        t += dt
        Δt = t - t_prev
        if neuron(u, t)[1]==true   # neuron fired
            i += 1 
            ISI[i] = Δt
            t_prev = t  
        elseif Δt > timeout
            @printf "interspike_intervals() timed out"
            return ISI[1:i]
        end
    end

    return ISI
end

# Neuron firing rate in response to stimulus u(t) up to time T
# via GLR filter with upper band limit f
function firingRate(neuron::Function, u::Function, T::Float64, f::Float64)

    spt = spiketimes(neuron, u, T)

    return GLR(spt, f, T)

end


#  construct torsion pendulum model (δ, δdot) = steinhausen(u,t), δ is cupula deflection
#  given input u(t) = head angular velocity or acceleration.
#  Steinhausen model: Θ.δ'' + Φ.δ' + Δ δ = Θ.wdot
#  If velocity==true (default) then u(t) is head angular velocity, else it's acceleration.
function make_torsion_pendulum(velocity::Bool=true, dt::Float64=DEFAULT_SIMULATION_DT)

    # Chinchilla canal parameters (Hullar)
    Theta = 1.53e-12   # coeff of q'', endolymph moment of inertia kg.m^2
    Phi = 4.23e-10     # coeff of q', viscous damping N.m.s/rad 
    Delta = 9.84e-11   # coeff of q, cupula stiffness N.m/rad 

    # Continuous-time model
    A = @SMatrix [0.0 1.0;
                 -Delta/Theta  -Phi/Theta]
    B = @SVector [0.0, 1.0]

    # Exact discrete-time step for piecewise-constant angular acceleration
    Ad = exp(A * dt)
    Bd = A \ ((Ad - I) * B)    

    # unpack scalars for fast execution in closure
    a11, a12 = Ad[1,1], Ad[1,2]
    a21, a22 = Ad[2,1], Ad[2,2]  
    b1, b2 = Bd[1], Bd[2]
    
    # canal state (endolymph displacement ~ cupula deflection)
    # initial (0.0, 0.0)
    δ = Ref(0.0)
    δdot = Ref(0.0)

    # canal/cupula model
    function torsionpendulum(u::Function, t::Float64)

        if velocity==true            # velocity input specified
            wdot = diffcd(u, t, dt)  # differentiate input
        else
            wdot = u(t)  
        end
        
        # state update
        δ_new       = a11*δ[] + a12*δdot[] + b1*wdot
        δdot_new    = a21*δ[] + a22*δdot[] + b2*wdot

        δ[] = δ_new
        δdot[] = δdot_new

        return (δ[], δdot[])  # return cupula state
    
    end

    # return closure
    return torsionpendulum
end

 