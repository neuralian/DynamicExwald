
# Returns closure that simulates 1 timestep (dt) of SLIF neuron
#                 and returns true if the neuron fired, false otherwise
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

    return SLIF_neuron
end