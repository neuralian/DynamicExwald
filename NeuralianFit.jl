# Neuralian fitting Functions
# MGP 2024
using Optimization
using OptimizationNLopt



# fit Exwald parameters to vector of interspike intervals
#   using maximum likelihood or minimum KL-divergence (Paulin, Pullar and Hoffman, 2024)
# uses NLopt
# returns (mu, lambda, tau)  (in units matching the input data, 
#                             e.g. seconds if intervals are specified in seconds )
function Fit_Exwald_to_ISI(ISI::Vector{Float64}, Pinit::Vector{Float64})


    # likelihood function
    grad = zeros(3)
    LHD = (param, grad) -> sum(log.(Exwaldpdf(param[1], param[2], param[3], ISI))) #- (param[1]^2 + param[3]^2)

    #@infiltrate

    optStruc = Opt(:LN_PRAXIS, 3)   # set up 3-parameter NLopt optimization problem

    optStruc.max_objective = LHD       # objective is to maximize likelihood

    optStruc.lower_bounds = [0.0, 0.0, 0.0]   # constrain all parameters > 0
    #optStruc.upper_bounds = [1.0, 25.0,5.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, pest, ret) = optimize(optStruc, Pinit)

    (maxf, abs.(pest), ret)

end

# fit closest to spontaneous
function Fit_Exwald_to_ISI(ISI::Vector{Float64}, spont::Vector{Float64}, Pinit::Vector{Float64})


    # likelihood function
    grad = zeros(3)
    #w = [0.0, 100.0, .0]
    LHD = (param, grad) -> sum(log.(Exwaldpdf(param[1]^2, param[2]^2, param[3]^2, ISI))) #- sum((w.*abs.(param-spont)./spont))

    #@infiltrate

    optStruc = Opt(:LN_NELDERMEAD, 3)   # set up 3-parameter NLopt optimization problem

    optStruc.max_objective = LHD       # objective is to maximize likelihood

    # optStruc.lower_bounds = [0.005, 0.01, 0.01]   # constrain all parameters > 0
    # optStruc.upper_bounds = [.2,50.0, 1.0]

    #optStruc.xtol_rel = 1e-12
    optStruc.xtol_rel = 1.0e-16

    Grad = zeros(3)  # dummy argument (uisng gradient free algorithm)
    (maxf, pest, ret) = optimize(optStruc, Pinit)

    println("pest: ",sqrt.(pest))

    (maxf, sqrt.(pest), ret)


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
        f::Float64, dt::Float64=0.01, N::Int64=0, Nburn::Int64=1)


    period = 1.0/f 
    w = 2πf


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
    Goodness = (param, grad) -> sum( (r .- sinewave(param,f,t)  ).^2)/length(t)

  #  optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem
    optStruc = Opt(:LN_PRAXIS,3)  # set up 3-parameter NLopt optimization problem
    
    NLopt.min_objective!(optStruc, Goodness)       # objective is to minimize goodness

    NLopt.lower_bounds!(optStruc, [-maximum(r), 0.0, -0.5])  # constrain all parameters > 0
    optStruc.upper_bounds = [maximum(r), (maximum(r)-minimum(r))/2.0, 0.5]

    #optStruc.xtol_rel = 1e-12
    NLopt.xtol_rel!(optStruc, 1.0e-16)



    (minf, pest, ret) = optimize(optStruc, Pinit)


    (pest, minf, ret)

end

# Fit sine wave of frequency F /Hz to firing rate samples
#   dt is sample interval
#   rate vector r should be trimmed before calling this function,
#   to give steady-state response (remove periods at start to allow burn-in) 
#   and no edge effects from rate estimator (remove last period)  
function fit_Sinewave_to_Firingrate(r::Vector{Float64}, f::Float64, dt::Float64)


    # sample times starting at t=dt
    N = length(r)
    t = collect((1:N)*dt)

    # parameters to be fitted are mean rate, amplitude and phase
    r0  = mean(r)
    a0 = (maximum(r)-r0)
    phi0 = -pi/2.
    Pinit = [r0, a0, phi0]

    grad = [0.0, 0.0, 0.0]   # required input to fitting code but not used
    # Goodness of fit is mean squared error
    Goodness = (param, grad) -> sum( (r .- sinewave(param,f,t)  ).^2)/N

    optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem
    #optStruc = Opt(:LN_PRAXIS,3)  # set up 3-parameter NLopt optimization problem
    
    NLopt.min_objective!(optStruc, Goodness)       # objective is to minimize goodness

    NLopt.lower_bounds!(optStruc, [-maximum(r), 0., -pi])  # constrain all parameters > 0
    optStruc.upper_bounds = [maximum(r), 2.0*maximum(r), pi]

    #optStruc.xtol_rel = 1e-12
    NLopt.xtol_rel!(optStruc, 1.0e-8)



    (minf, pest, ret) = optimize(optStruc, Pinit)


    (pest, minf, ret)

end


# fit Exponential distribution to normalized histogram data, P_bin = bin_count*binwidth
# returns parameter τ
function fit_Exponential(bin_centre::Vector{Float64}, P_bin::Vector{Float64}; max_iter=100, tol=1e-6)

    @assert isapprox(sum(P_bin), 1.0, atol = 1.0e-6)  "probabilities must sum to 1"

    binwidth = bin_centre[2] - bin_centre[1]
    
    # initial estimate is average interval length
    τ = sum(bin_centre .* P_bin)
    pInit = [τ]
    
    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence
    # nb max.() kluge because pdf.() returns < 0.0 for some params (not my bug)
    dist = param -> 
        pdf.(Exponential(param...), bin_centre)/sum(pdf.(Exponential(param...), bin_centre))
    KL_divergence = (param, grad) -> KLD(bin_centre, P_bin, dist(param))

    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
    optStruc = Opt(:LN_NELDERMEAD, length(pInit)) # :NL_PRAXIS
    NLopt.min_objective!(optStruc, KL_divergence)      
    NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    NLopt.xtol_rel!(optStruc, 1.0e-12)

    #@infiltrate

    (minf, pest, ret) = optimize(optStruc, pInit)
    #println(pest, "    ", pInit)

    #@infiltrate
    return pest, minf
end


# fit Wald distribution to normalized histogram data, P_bin = bin_count*binwidth
# returns parameters μ, λ
function fit_Wald(bin_centre::Vector{Float64}, P_bin::Vector{Float64}; max_iter=100, tol=1e-6)

    @assert isapprox(sum(P_bin), 1.0, atol = 1.0e-6)  "probabilities must sum to 1"

    binwidth = bin_centre[2] - bin_centre[1]
    
    # initial estimates from summary stats
    μ = sum(bin_centre .* P_bin)
    V = sum(P_bin.*bin_centre.^2) - μ^2  
    λ = μ^3 / V
    pInit = [μ, λ]
    
    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence
    # nb max.() kluge because pdf.() returns < 0.0 for some params (not my bug)
    dist = param -> max.(0.0, 
        pdf.(InverseGaussian(param...), bin_centre)/sum(pdf.(InverseGaussian(param...), bin_centre)))
    KL_divergence = (param, grad) -> KLD(bin_centre, P_bin, dist(param))

    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
    optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    NLopt.min_objective!(optStruc, KL_divergence)      
    NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    NLopt.xtol_rel!(optStruc, 1.0e-12)

    (minf, pest, ret) = optimize(optStruc, pInit)
    #println(pest, "    ", pInit)

    #@infiltrate
    return pest, minf
end


# fit Exwald distribution to histogram data
# returns parameters μ, λ, τ
function fit_Exwald(bin_centre::Vector{Float64}, bin_count::Vector{Float64}; max_iter=100, tol=1e-6)

    binwidth = bin_centre[2] - bin_centre[1]
    P_bin = bin_count/sum(bin_count)   # 
    
    # initial estimates from summary stats
    # divide average interval into halves avg = μ + τ
    avg = sum(bin_centre .* P_bin)
    V = sum(P_bin.*bin_centre.^2) - avg^2  # data variance = E[x^2] - E[x]^2
    w = min(sqrt(V)/avg, 0.95)
    τ = w*avg
    μ = (1.0-w)*avg
    # set initial λ>>λ_w because if λ is large there is a local deep pocket in the objective function  
    # around its value (corresponding Exwald with steep leading edge). Local seach algorithms
    # have trouble finding small holes in the objective function unless they start there.  
    λ = 25.0*μ^3 / (V*(1-w)^2)
    pInit = [μ, λ, τ] 
    println(pInit)
    LB = [0., 0., 0.]  # lower bounds
    UB = [avg, 10.0*λ, avg ]    # upper bounds
    
    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence
    dist = param -> Exwaldpdf(abs(param[1]), abs(param[2]), abs(param[3]), bin_centre, true)
    KL_divergence = (param, grad) -> KLD(bin_centre, P_bin, dist(param))

    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
 #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
    # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    # NLopt.min_objective!(optStruc, KL_divergence)      
    # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    # NLopt.xtol_abs!(optStruc, 1.0e-12)
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_PRAXIS(), reltol = 1.0e-12)
    #println(pest, "    ", pInit)

    #@infiltrate
    return sol.u , sol.objective
end
