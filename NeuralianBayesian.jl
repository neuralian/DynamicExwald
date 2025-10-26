# tools for computing likelihoods and posteriors in state space
# cupula state space (δ, δ') is isomorphic to head kinematic state space (ω, ̇ω)
# empirical evidence indicates that semicircular canal afferents transmit information
#   about ω and ̇ω, i.e. kinematic state of cupula indicates kinematic state of head (re that d.f.)


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


