# Neuralian fitting Functions
# MGP 2024
using Optimization
using OptimizationNLopt

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
# f_bin is probability density at sample point, probability = f_bin*binwidth
function fit_Exwald(bin_centre::Vector{Float64}, f_bin::Vector{Float64}; 
                   pInit::Tuple{Float64, Float64, Float64} = (NaN, NaN, NaN), max_iter=100, tol=1e-6)

    bw = bin_centre[2] - bin_centre[1]
    
    # initial estimates from summary stats
    # divide average interval into halves avg = μ + τ
    if any(isnan.(pInit))
        avg = sum(bw*bin_centre .* f_bin)
        V = sum(bw*f_bin.*bin_centre.^2) - avg^2  # data variance = E[x^2] - E[x]^2
        cv = sqrt(V)/avg
        w = min(cv, 0.95)
        τ = w*avg
        μ = (1.0-w)*avg
        λ = μ/cv^2
        pInit = [μ, λ, τ]
    end



 #   println(pInit)
    
    # parameter bounds 
    avg = pInit[1]+pInit[3]
    LB = [log(1.0e-4), log(1.0e-3), log(1.0e-6)]  # lower bounds (empirical, from PP&H data)
    UB = [log(10.0*avg), log(100.0*pInit[2]), log(10.0*avg) ]    # upper bounds
       
    # initial parameters as vector
    pInit = log.(collect(pInit))
    println(" ", pInit)

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence
    dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
    KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)


    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
 #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
    # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    # NLopt.min_objective!(optStruc, KL_divergence)      
    # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    # NLopt.xtol_abs!(optStruc, 1.0e-12)
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_NELDERMEAD(), reltol = 1.0e-9)
 
  #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

    #@infiltrate
    return tuple(exp.(sol.u)...) , sol.objective
end

# fit Exwald distribution to interval data
# returns parameters μ, λ, τ
function sfit_Exwald(ISI::Vector{Float64};
            pInit::Tuple{Float64, Float64, Float64} = (NaN, NaN, NaN), max_iter=100, tol=1e-6)
    
    # initial estimates from summary stats
    # divide average interval into halves avg = μ + τ
    if any(isnan.(pInit))  # if pInit was not specified or is invalid
        avg = mean(ISI)
        V = mean(ISI.^2) - avg^2  # data variance = E[x^2] - E[x]^2
        cv = sqrt(V)/avg
        w = min(cv, 0.95)
        τ = w*avg
        μ = (1.0-w)*avg
        λ = μ/cv^2
        pInit = [μ, λ, τ]
    end # otherwise pInit was passed as argument

    # parameter bounds 
    avg = pInit[1]+pInit[3]
    LB = [log(1.0e-4), log(1.0e-3), log(1.0e-6)]  # lower bounds (empirical, from PP&H data)
    UB = [log(10.0*avg), log(100.0*pInit[2]), log(10.0*avg) ]    # upper bounds
       
    # initial parameters as vector
    pInit = log.(collect(pInit))
    println(" ", pInit)

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence from data to model
    # dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
    # KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)
    KL_divergence = (param, grad) -> sKLD(ISI, d->Exwaldpdf(exp.(param)..., d) )


    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
 #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
    # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    # NLopt.min_objective!(optStruc, KL_divergence)      
    # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    # NLopt.xtol_abs!(optStruc, 1.0e-12)
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_NELDERMEAD(), reltol = 1.0e-9)
 
  #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

    #@infiltrate
    return tuple(exp.(sol.u)...) , sol.objective
end


# fit Exwald parameters to SLIF neuron =
# Ornstein-Uhlenbeck first passage time model tau.dx/dt + x = N(v0, sigma), barrier==1
# given SLIF parameters  (v0, sigma, tau) 
# Returns Exwald parameters (μ, λ, τ) fitted to SLIF neuron intervals
# pInit =  initial parameters
# meanrange = valid range of mean ISI, returns (NaN, NaN, NaN) outside this range
# (allows neurons with unrealistically high or low firing rates to be skipped during searches)
function fit_Exwald_to_SLIF(SLIFparam::Tuple{Float64, Float64, Float64}, colour = 0.1, N::Int64=10000, 
                pInit::Tuple{Float64, Float64, Float64}=(NaN, NaN, NaN), 
                meanrange::Tuple{Float64, Float64} = (0.0, 1.0), timeout::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    SLIFneuron = make_SLIF_neuron(SLIFparam, colour, dt) 
    ISI = interspike_intervals(SLIFneuron, t->0.0, N) 
    if length(ISI)<N  || mean(ISI)<meanrange[1] || mean(ISI)>meanrange[2] 
        # neuron model does not generate plausible intervals, skip it
        return (NaN, NaN, NaN)  
    end

#   #  println(", ", any(isnan.(f_bin)), ", ", any(isinf.(f_bin)), ", ", maximum(f_bin))
#     EXWparam, ob = sfit_Exwald(ISI, pInit=pInit)

#     return EXWparam, ob

    avg = mean(ISI)
   
  #  @infiltrate

    # initial estimates from summary stats
    # divide average interval into halves avg = μ + τ
    if any(isnan.(pInit))  # if pInit was not specified or is invalid
        V = mean(ISI.^2) - avg^2  # data variance = E[x^2] - E[x]^2
        if V<1.0e-8
            println(V)
            return (NaN, NaN, NaN)
        end
        cv = sqrt(V)/avg
        w = min(cv/8.0, 0.95)
        τ = w*avg
        μ = (1.0-w)*avg
        λ = μ/cv^2
        pInit = [μ, λ, τ]
    end 

    # if (avg > .1) || (avg < .01)  # rate > 100 or < 10
    #     return (NaN, NaN, NaN), NaN
    # else

    # parameter bounds 
    LB = [log(1.0e-5), log(1.0e-4), log(1.0e-7)]  # lower bounds (empirical, from PP&H data)
    UB = [log(100.0*avg),  log(100.0*pInit[2]), log(100.0*avg) ]    # upper bounds
       
    # initial parameters as vector
    pInit = log.(collect(pInit))
    #println(" ", pInit)

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence from data to model
    # dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
    # KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)
    KL_divergence = (param, grad) -> sKLD(ISI, d->Exwaldpdf(exp.(param)..., d) )


    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
 #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
    # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    # NLopt.min_objective!(optStruc, KL_divergence)      
    # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    # NLopt.xtol_abs!(optStruc, 1.0e-12)
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_PRAXIS(), abstol = 1.0e-12)
 
  #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

    #@infiltrate
    # end

    return tuple(exp.(sol.u)...)

end

# fit Exwald parameters to fractional dynamic SLIF model
# SLIF parameters are (μ, λ, τₘ) and q, where τₘ is mean-reverting time constant 
#                                      (= membrane time constant in LIF neuron)
# Exwald parameters are (μ, λ, τₓ).
# Fitting is done by generating N intervals from OU neuron & using fit_Exwald. 
# pInit = specified initial parameters
# meanrange = valid range of mean ISI, returns (NaN, NaN, NaN) outside this range
function fit_Exwald_to_qSLIF(SLIFparam::Tuple{Float64, Float64, Float64}, q::Float64 = 0.0, N::Int64=10000, 
                pInit::Tuple{Float64, Float64, Float64}=(NaN, NaN, NaN), 
                meanrange::Tuple{Float64, Float64} = (0.0, 1.0), timeout::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    qSLIFneuron = make_fractional_SLIF_neuron(SLIFparam, q, dt) 
    ISI = interspike_intervals(qSLIFneuron, t->0.0, N) 
    if length(ISI)<N  || mean(ISI)<meanrange[1] || mean(ISI)>meanrange[2] 
    # neuron does not generate plausible intervals
        return (NaN, NaN, NaN), NaN   # don't even try 
    end

#   #  println(", ", any(isnan.(f_bin)), ", ", any(isinf.(f_bin)), ", ", maximum(f_bin))
#     EXWparam, ob = sfit_Exwald(ISI, pInit=pInit)

#     return EXWparam, ob

    avg = mean(ISI)
   
    # initial estimates from summary stats
    # divide average interval into halves avg = μ + τ
    if any(isnan.(pInit))  # if pInit was not specified or is invalid
        V = mean(ISI.^2) - avg^2  # data variance = E[x^2] - E[x]^2
        cv = sqrt(V)/avg
        w = min(cv/8.0, 0.75)
        τ = w*avg
        μ = (1.0-w)*avg
        λ = μ/cv^2
        pInit = [μ, λ, τ]
    end 

    # if (avg > .1) || (avg < .01)  # rate > 100 or < 10
    #     return (NaN, NaN, NaN), NaN
    # else

    # parameter bounds 
    LB = [log(1.0e-5), log(1.0e-4), log(1.0e-7)]  # lower bounds (empirical, from PP&H data)
    UB = [log(100.0*avg),  log(100.0*pInit[2]), log(100.0*avg) ]    # upper bounds
       
    # initial parameters as vector
    pInit = log.(collect(pInit))
    println(" ", pInit)

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is mean squared error
    # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

    # Goodness of fit is Kullback-Leibler divergence from data to model
    # dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
    # KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)
    KL_divergence = (param, grad) -> sKLD(ISI, d->Exwaldpdf(exp.(param)..., d) )


    #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

    # set up optimization
 #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
    # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
    # NLopt.min_objective!(optStruc, KL_divergence)      
    # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
    # NLopt.xtol_abs!(optStruc, 1.0e-12)
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_PRAXIS(), abstol = 1.0e-12)
 
  #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

    #@infiltrate
    # end

    return tuple(exp.(sol.u)...), sol.objective

end