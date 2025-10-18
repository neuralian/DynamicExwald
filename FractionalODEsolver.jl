DEFAULT_SIMULATION_DT = 1.0e-5

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

# Closure defining state update function for Steinhausen model y'' + A y' + B Dq y = u(t)
# with visco-elastic cupular restoring force modeled by fractional derivative Dq = d^q/dt^q
# Using Oustaloup approximation to Dq over specified frequency band.
# The augmented state includes auxiliary variables for the Oustaloup approximation.
# Initial state is [y0, y'(0), 0, 0, ..., 0] (M zeros for auxiliary states).
function make_fractional_Steinhausen_stateUpdate_fcn(
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

    # Augmented state: u = [y, y', v1, ..., vM]
    x = [y0; yp0; zeros(M)]
    du = zeros(length(x))

    # State update function
    function update(u::Float64)

        y =  x[1]
        dy = x[2]
        vs = @view x[3:end]
        
        # Approximate D^q u 
        if (q==0.0) 
            approx_dq = u
        else
            approx_dq = K * u
            for i in 1:M
                approx_dq += residues[i] * vs[i]
            end
        end
        
        # "ordinary" state updates
        du[1] = dy
        du[2] = approx_dq - A * dy - B * y
        
        # Auxiliary state updates
        for i in 1:M
            du[2 + i] = -p_i[i] * vs[i] + u
        end

        # Euler integration
        for i in 1:length(x)
            x[i] += du[i] * dt
        end

        return x[1]  # return cupula deflection
    
    end

    return update
end
    
# Example usage
q = -0.5  # fractional order
w = 2.0  # frequency of input rad/s
T = 12.0
dt = DEFAULT_SIMULATION_DT
t = 0:dt:T
x = sin.(w .* t)  # input angular velocity (rad/s)
 d = zeros(length(t))

 FSS_update = make_fractional_Steinhausen_stateUpdate_fcn(q, 0., 0.)

 for i in 1:length(t)
     d[i] = FSS_update(x[i])
 end