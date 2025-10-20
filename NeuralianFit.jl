# Neuralian fitting Functions
# MGP 2024
using NLopt



# fit Exwald parameters to vector of interspike intervals
#   using maximum likelihood or minimum KL-divergence (Paulin, Pullar and Hoffman, 2024)
# uses NLopt
# returns (mu, lambda, tau)  (in units matching the input data, 
#                             e.g. seconds if intervals are specified in seconds )
function Fit_Exwald_to_ISI(ISI::Vector{Float64}, Pinit::Vector{Float64})


    # likelihood function
    grad = zeros(3)
    LHD = (param, grad) -> sum(log.(Exwaldpdf(param[1], param[2], param[3], ISI))) - (param[1]^2 + param[3]^2)

    #@infiltrate

    optStruc = Opt(:LN_PRAXIS, 3)   # set up 3-parameter NLopt optimization problem

    optStruc.max_objective = LHD       # objective is to maximize likelihood

    optStruc.lower_bounds = [0.0, 0.0, 0.0]   # constrain all parameters > 0
    #optStruc.upper_bounds = [1.0, 25.0,5.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, pest, ret) = optimize(optStruc, Pinit)


end

# fit closest to spontaneous
function Fit_Exwald_to_ISI(ISI::Vector{Float64}, spont::Vector{Float64}, Pinit::Vector{Float64})


    # likelihood function
    grad = zeros(3)
    #w = [0.0, 100.0, .0]
    LHD = (param, grad) -> sum(log.(Exwaldpdf(param[1], param[2], param[3], ISI))) #- sum((w.*abs.(param-spont)./spont))

    #@infiltrate

    optStruc = Opt(:LN_PRAXIS, 3)   # set up 3-parameter NLopt optimization problem

    optStruc.max_objective = LHD       # objective is to maximize likelihood

    optStruc.lower_bounds = [0.0, 0.0, 0.0]   # constrain all parameters > 0
    #optStruc.upper_bounds = [1.0, 25.0,5.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, pest, ret) = optimize(optStruc, Pinit)


end

function Fit_scaled_Exwald_to_ISI(ISI::Vector{Float64}, Pinit::Vector{Float64}, s::Float64)

    # scaling
    Pinit[3] = Pinit[3] / s

    # likelihood function
    grad = zeros(3)
    LHD = (param, grad) -> sum(log.(scaled_Exwaldpdf(param[1], param[2], param[3], ISI, s)))

    #@infiltrate

    optStruc = Opt(:LN_PRAXIS, 3)   # set up 3-parameter NLopt optimization problem



    optStruc.max_objective = LHD       # objective is to maximize likelihood

    optStruc.lower_bounds = [0.0, 0.0, 0.0]   # constrain all parameters > 0
    #optStruc.upper_bounds = [1.0, 25.0,5.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, pest, ret) = optimize(optStruc, Pinit)

    pest[3] = pest[3] * s

    (maxf, pest, ret)

end

# Fit sine wave of frequency F /Hz to spike train (rate) 
#   Nburn = number of cycles to drop at start of spiketrain
#   N = number of periods to fit. 
#       If N=0 (default), then N is the number of full cycles in the data after Nburn cycles
#   sd is GLR filter sd 
#   dt is the GLR sample resolution, default .01 corresponds to max sine frequnecy (Nyquist) 50Hz
# 
function Fit_Sinewave_to_Spiketrain(spiketime::Vector{Float64}, 
        F::Float64, dt::Float64=0.01, N::Int64=0, Nburn::Int64=1)


    period = 1.0/F 
    if N==0
        N = Int(floor((spiketime[end]-Nburn*period)/period))  # number of full cycles in data after burn-in
    end

    T = (Nburn + N)*period  # end of fitting interval
    #println((Nburn+N)*period, T)

    # firing rate (all data)
    (t, r) = GLR(spiketime, period/16.0, dt)

    # find indices for fitting interval
    iFit = findall(s -> s>=Nburn*period && s<=T, t)


    # select data to fit, reset t=0
    t = t[iFit].-t[iFit[1]]
    r = r[iFit]

    Pinit = [mean(r),0.1*(maximum(r)-mean(r))/sqrt(2.0), 0.1]

    grad = [0.0, 0.0, 0.0]   # required but not used
    # Goodness of fit is sum of squared residual
 #   Goodness = (param, grad) -> sum( (r .- (param[1] .+ param[2]*sin.(2.0*π*(F*t.-param[3])))  ).^2) 
    Goodness = (param, grad) -> sum( (r .- sinewave(param,F,t)  ).^2)/length(t)

  #  optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem
    optStruc = Opt(:LN_PRAXIS,3)  # set up 3-parameter NLopt optimization problem
    
    NLopt.min_objective!(optStruc, Goodness)       # objective is to minimize goodness

    NLopt.lower_bounds!(optStruc, [-maximum(r), 0.0, -0.5])  # constrain all parameters > 0
    optStruc.upper_bounds = [maximum(r), (maximum(r)-minimum(r))/2.0, 0.5]

    #optStruc.xtol_rel = 1e-12
    NLopt.xtol_rel!(optStruc, 1.0e-16)



    (minf, pest, ret) = optimize(optStruc, Pinit)


    (minf, pest, ret)

end

# Fit sine wave of frequency F /Hz to firing rate samples
#   dt is sample interval
# 
function Fit_Sinewave_to_Firingrate(r::Vector{Float64}, f::Float64, dt::Float64)


    # sample times starting at t=dt
    N = length(r)
    t = collect((1:N)*dt)

    # parameters to be fitted are mean rate, amplitude and phase
    r0  = mean(r)
    a0 = (maximum(r)-r0)/sqrt(2.0)
    phi0 = 0.0
    Pinit = [r0, a0, phi0]

    grad = [0.0, 0.0, 0.0]   # required input to fitting code but not used
    # Goodness of fit is mean squared error
    Goodness = (param, grad) -> sum( (r .- sinewave(param,f,t)  ).^2)/N

  #  optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem
    optStruc = Opt(:LN_PRAXIS,3)  # set up 3-parameter NLopt optimization problem
    
    NLopt.min_objective!(optStruc, Goodness)       # objective is to minimize goodness

    NLopt.lower_bounds!(optStruc, [-maximum(r), 0.0, -.5])  # constrain all parameters > 0
    optStruc.upper_bounds = [maximum(r), 2.0*maximum(r), .5]

    #optStruc.xtol_rel = 1e-12
    NLopt.xtol_rel!(optStruc, 1.0e-16)



    (minf, pest, ret) = optimize(optStruc, Pinit)


    (minf, pest, ret)

end
