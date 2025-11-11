# tools for computing likelihoods and posteriors in state space
# cupula state space (δ, δ') is isomorphic to head kinematic state space (ω, ̇ω)
# empirical evidence indicates that semicircular canal afferents transmit information
#   about ω and ̇ω, i.e. kinematic state of cupula indicates kinematic state of head (re that d.f.)

# return closure to infer the posterior probability that spikes are generated  
#  by an Exwald neuron with parameters EXWparam, sequentially at spike times.
function make_Bayesian_dendrite(EXWparam::Tuple{Float64, Float64, Float64})

    previous_spiketime = 0.0        # time of previous spike
    p = 1.0                         # membrane potential (Bayes numerator), posterior probability when normalized

    function dendrite_update(spiketime::Float64)

        interval = spiketime - previous_spiketime
        previous_spiketime = spiketime
        if interval > 0.0   # to get posterior without updating, call with interval <= 0.0
            # Bayes numerator, likelihood x prior
            p = Exwaldpdf(EXWparam..., interval)*p
        end

        # p is probability when normalized
        return p

    end

    # Evidence is Sum(p) over all dendrites in all secondary neurons
    function dendrite_renormalize(Evidence::Float64)

        p /= Evidence

        # posterior probability 
        return p
    end

    # get dendrite numerator 
    function dendrite_getp()

        return p

    end


    return dendrite_update, dendrite_renormalize

end

# Construct a Bayesian secondary neuron as a collection of Bayesian dendrites (compartments)
#   EXWparam[i] contains Exwald parameters (likelihood function) for the ith dendrite 
#   a[i] is index specifying which neuron in the afferent nerve projects to ith dendrite of this neuron.
#   (see make_afferent_nerve in Neuralian_models.jl)
function make_Bayesian_neuron(EXWparam::Vector{Tuple{Float64, Float64, Float64}}, a::Vector{Int64})

    N = length(EXWparam)

    # the neuron is an array of dendrites
    dendrite_update = Vector{Function}(undef, N)   # update function for each dendrite
    dendrite_renormalize = Vector{Function}(undef, N)   # update function for each dendrite
   
    for i in 1:N
        dendrite_update[i], dendrite_renormalize[i] = make_Bayesian_dendrite(EXWparam[i])
    end

    # update Bayes numerator at spiketime of pth primary afferent
    function Bayesian_neuron_update(spiketime::Float64, j::Int64)

        # jth afferent projects to ith dendrite on this neuron
        # if i is empty then the jth afferent does not project to this neuron 
        i = findall(j.==a)[]

        p = 0.0
        if isempty(i)
            p += dendrite_getp[i]()                 # get Bayes numerator 
        else
            p += dendrite_update[i](spiketime)   # update and get Bayes numerator 
        end

        return p
    end

    function Bayesian_neuron_renormalize(Evidence::Float64)

        p = 0.0

        for i in 1:N
            p += dendrite_renormalize[i](Evidence)   # normalize & return posterior probability
        end 

        # posterior probability at this location (state) in the neuron map
        return p
    end

    return Bayesian_neuron_update, Bayesian_neuron_renormalize

end


# A Bayesian neuron map is an MxN array of Bayesian neurons 
# Closure updates the posterior distribution over the map given the spike time of an afferent neuron.
# It must be called for each spike in each afferent in temporal order.
# EXWparam is MxN array where EXWparam[i,j] is a vector whose elements are 
#     Exwald parameters (μ, λ, τ) for each dendrite on the neuron at map location [i,j].  
#     The number of dendrites on the neuron is determined by the length of this vector.
# afferentProjection is an MxN array whose [i,j]th element is a vector whose kth element gives the index 
#     of the afferent neuron that projects to the kth dendrite of the [i,j]th map neuron.
#     See make_afferent_nerve() in Neuralian_models.jl.
#     The length of afferentProjection[i,j] must equal the number of dendrites (compartments) in the 
#     map neuron at [i,j].
# 1D map can be specified using N- or M-vectors instead of 1xN or Mx1 matrices 
#    (via a convenience function defined below that casts the parameters to matrices then call this function)
#  You can get the current posterior map without updating by calling the update function with a negative spiketime) 
function make_Bayesian_neuron_map(
    EXWparam::Matrix{Vector{Tuple{Float64, Float64, Float64}}}, 
    afferentProjection::Matrix{Vector{Int64}}                    )

    # check dimensions and convert M-vector inputs (1D maps) to 1xM Matrices
    M, N = size(EXWparam)
    _M, _N = size(afferentProjection)
    @assert (M==_M) && (N==_N)  "Afferent projection matrix must be the same size as the secondary neuron map"

    # NB EXWparam[i,j] contains a vector of Exwald parameters (μₖ, λₖ, τₖ) defining likelihood
    #    functions for each dendrite in the Bayesian neuron at location [i,j] in the map. 
    #    afferentProjection[i,j] contains a vector aₖ specifying which afferents in the nerve 
    #    project to each dendrite of this map neuron.


    Evidence = Float64(M*N)
    p = Matrix{Float64}(undef, M, N)  #  probability distribution over the map

    # construct array of Bayesian secondary neurons
    # initialize prior probability to 1/Evidence in each dendrite
    map_neuron = Array{Function}(undef, M, N) 
    renormalize = Array{Function}(undef, M, N)
    for i = 1:M
        for j = 1:N
            map_neuron[i,j], renormalize[i,j] = make_Bayesian_neuron(EXWparam[i,j], afferentProjection[j])
            p[i,j] = renormalize[i,j](Evidence)
        end
    end

    # update map at spiketime of kth afferent neuron 
    function Bayesian_map_update(spiketime::Float64, k::Int64)

        # update membrane potentials (numerators)
        # nb Evidence accumulates from all secondary neurons, even if not connected to kth primary neuron
        Evidence = 0.0
        for i in 1:M
            for j in 1:N
                Evidence += map_neuron[i,j](spiketime, k)
            end
        end

        # normalize posterior distribution (divide by sum of numerators)
        # nb calling Bayes_normalizer for each element instead of simply dividing by Evidence here
        # because that updates the internal state (belief) of each decoder, allows sequential inference
        for i in 1:M
            for j in 1:N
                p[i,j] = renormalize[i,j](Evidence)
            end
        end

        # return posterior probability distribution over the map
        return p

    end

    return Bayesian_map_update

end

# convenience function for 1D map.
# Converts vector inputs to matrices and dispatches to the matrix version.
function make_Bayesian_neuron_map(
    EXWparam::Vector{Vector{Tuple{Float64, Float64, Float64}}}, 
    afferentProjection::Vector{Vector{Int64}}                       )

    # convert M-vectors to 1xM Matrices
    update = make_Bayesian_neuron_map(reshape(EXWparam, 1, :), reshape(afferentProjection, 1, :))

    return update

end


# construct an array to hold histogram/pdf of states (ω, ̇ω) visited by BLG stimulus 
# Δω = velocity resolution of map, radius = radius of map wrt rms stimulus, default 2.0 (=2 x s.d.)
function init_stateMap(blgParams, Δω = 1.0, radius::Float64=2.0)

    (flower, fupper, rms ) = blgParams

    # resolution in α = ̇ω direction required for a square map 
    # (scale proportional to ratio of rms_α to rms_ω )
    αrms = blg_derivative_RMS(blgParams)
    Δα = Δω*αrms/rms

    ωMax = radius*rms         # map velocities up to ±2 s.d. from 0
    αMax = radius*αrms        # ditto accelerations 

    # square matrix to hold stimulus histogram (map of states visited by the stimulus)
    # the map spans ±ωMax ±αMax with grid elements of size (Δw, Δα)
    M = max(Int(ceil(ωMax/Δω)), Int(ceil(αMax/Δα)) ) 
    
    # return stateMap, a tuple containing an empty array of the required size,  
    # map resolution (pixel diameter) in each direction, and map extent (ωMax, αMax)
    stateMap = (zeros(Float64, 2*M+1, 2*M+1), (Δω, Δα), (ωMax, αMax) )

end


# function to insert a value in a stateMap cell 
# warn (or not) if state is not within bounds of map
function map_insert(stateMap::Tuple{Matrix{Float64}, Tuple{Float64, Float64}}, 
            state::Tuple{Float64, Float64}, value::Float64, warn::Bool=true)  

    (map, (Δω, Δα) ) = stateMap

    M = Int((size(map, 2) - 1)/2)
    
    i = 1 + M +  Int(floor(state[1]/Δω)) # 1 if state[1]==-ωMax, M+1 if state[1] = 0., 2*M+1 if state = ωMax
    j = 1 + M +  Int(floor(state[2]/Δα)) # simly

    if checkbounds(Bool, map, i, j)  # assign value to state if the state is in the map
        map[i,j] = value
    elseif warn
        println(@sprintf "Warning: state ( %.2f, %.2f ) is off the map" state...)
    end

end

# add value to current value for state
function map_add(stateMap::Tuple{Matrix{Float64}, Tuple{Float64, Float64}, Tuple{Float64, Float64}}, 
            state::Tuple{Float64, Float64}, value::Float64, warn::Bool=true)   

    (map, (Δω, Δα) ) = stateMap

    M = Int((size(map, 2) - 1)/2)
    
    i = 1 + M +  Int(floor(state[1]/Δω)) # 1 if state[1]==-ωMax, M+1 if state[1] = 0., 2*M+1 if state = ωMax
    j = 1 + M +  Int(floor(state[2]/Δα)) # simly
    if checkbounds(Bool, map, i, j)  # check that the state is in the map
        map[i,j] += value
    elseif warn
        println(@sprintf "Warning: state ( %.2f, %.2f ) is off the map" state...)
    end

end

# return map value at state
function map_read(stateMap::Tuple{Matrix{Float64}, Tuple{Float64, Float64}}, 
            state::Tuple{Float64, Float64}, warn::Bool=true)   

    (map, (Δω, Δα) ) = stateMap
    
    M = Int((size(map, 2) - 1)/2)

    i = 1 + M +  Int(floor(state[1]/Δω)) # 1 if state[1]==-ωMax, M+1 if state[1] = 0., 2*M+1 if state = ωMax
    j = 1 + M +  Int(floor(state[2]/Δα)) # simly

    if checkbounds(Bool, map, i, j)  # check that the state is in the map
        value = map[i,j] 
    elseif warn
        println(@sprintf "Warning: state ( %.2f, %.2f ) is off the map" state...)
    end

    return value
end


