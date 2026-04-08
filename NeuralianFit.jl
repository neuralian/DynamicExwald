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


# # fit Wald distribution to normalized histogram data, P_bin = bin_count*binwidth
# # returns parameters μ, λ
# function fit_Wald(bin_centre::Vector{Float64}, P_bin::Vector{Float64}; max_iter=100, tol=1e-6)

#     @assert isapprox(sum(P_bin), 1.0, atol = 1.0e-6)  "probabilities must sum to 1"

#     binwidth = bin_centre[2] - bin_centre[1]
    
#     # initial estimates from summary stats
#     μ = sum(bin_centre .* P_bin)
#     V = sum(P_bin.*bin_centre.^2) - μ^2  
#     λ = μ^3 / V
#     pInit = [μ, λ]
    
#     grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

#     # Goodness of fit is mean squared error
#     # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

#     # Goodness of fit is Kullback-Leibler divergence
#     # nb max.() kluge because pdf.() returns < 0.0 for some params (not my bug)
#     dist = param -> max.(0.0, 
#         pdf.(InverseGaussian(param...), bin_centre)/sum(pdf.(InverseGaussian(param...), bin_centre)))
#     KL_divergence = (param, grad) -> KLD(bin_centre, P_bin, dist(param))

#     #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

#     # set up optimization
#     optStruc = Opt(:LN_PRAXIS, length(pInit)) 
#     NLopt.min_objective!(optStruc, KL_divergence)      
#     NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
#     NLopt.xtol_rel!(optStruc, 1.0e-12)

#     (minf, pest, ret) = optimize(optStruc, pInit)
#     #println(pest, "    ", pInit)

#     #@infiltrate
#     return pest, minf
# end


# # fit Exwald distribution to histogram data
# # returns parameters μ, λ, τ
# # f_bin is probability density at sample point, probability = f_bin*binwidth
# function fit_Exwald(bin_centre::Vector{Float64}, f_bin::Vector{Float64}; 
#                    pInit::Tuple{Float64, Float64, Float64} = (NaN, NaN, NaN), max_iter=100, tol=1e-6)

#     bw = bin_centre[2] - bin_centre[1]
    
#     # initial estimates from summary stats
#     # divide average interval into halves avg = μ + τ
#     if any(isnan.(pInit))
#         avg = sum(bw*bin_centre .* f_bin)
#         V = sum(bw*f_bin.*bin_centre.^2) - avg^2  # data variance = E[x^2] - E[x]^2
#         cv = sqrt(V)/avg
#         w = min(cv, 0.95)
#         τ = w*avg
#         μ = (1.0-w)*avg
#         λ = μ/cv^2
#         pInit = [μ, λ, τ]
#     end



#  #   println(pInit)
    
#     # parameter bounds 
#     avg = pInit[1]+pInit[3]
#     LB = [log(1.0e-4), log(1.0e-3), log(1.0e-6)]  # lower bounds (empirical, from PP&H data)
#     UB = [log(10.0*avg), log(100.0*pInit[2]), log(10.0*avg) ]    # upper bounds
       
#     # initial parameters as vector
#     pInit = log.(collect(pInit))
#     println(" ", pInit)

#     grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

#     # Goodness of fit is mean squared error
#     # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

#     # Goodness of fit is Kullback-Leibler divergence
#     dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
#     KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)


#     #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

#     # set up optimization
#  #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
#     # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
#     # NLopt.min_objective!(optStruc, KL_divergence)      
#     # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
#     # NLopt.xtol_abs!(optStruc, 1.0e-12)
#     f = OptimizationFunction(KL_divergence)
#     Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
#     sol = solve(Prob, NLopt.LN_NELDERMEAD(), reltol = 1.0e-9)
 
#   #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

#     #@infiltrate
#     return tuple(exp.(sol.u)...) , sol.objective
# end

# # fit Exwald distribution to interval data
# # returns parameters μ, λ, τ
# function sfit_Exwald(ISI::Vector{Float64};
#             pInit::Tuple{Float64, Float64, Float64} = (NaN, NaN, NaN), max_iter=100, tol=1e-6)
    
#     # initial estimates from summary stats
#     # divide average interval into halves avg = μ + τ
#     if any(isnan.(pInit))  # if pInit was not specified or is invalid
#         avg = mean(ISI)
#         V = mean(ISI.^2) - avg^2  # data variance = E[x^2] - E[x]^2
#         cv = sqrt(V)/avg
#         w = min(cv, 0.95)
#         τ = w*avg
#         μ = (1.0-w)*avg
#         λ = μ/cv^2
#         pInit = [μ, λ, τ]
#     end # otherwise pInit was passed as argument

#     # parameter bounds 
#     avg = pInit[1]+pInit[3]
#     LB = [log(1.0e-4), log(1.0e-3), log(1.0e-6)]  # lower bounds (empirical, from PP&H data)
#     UB = [log(10.0*avg), log(100.0*pInit[2]), log(10.0*avg) ]    # upper bounds
       
#     # initial parameters as vector
#     pInit = log.(collect(pInit))
#     println(" ", pInit)

#     grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

#     # Goodness of fit is mean squared error
#     # SumSquaredError = (param, grad) -> sum( (bin_count - pdf.(InverseGaussian(param...), bincentre)).^2)

#     # Goodness of fit is Kullback-Leibler divergence from data to model
#     # dist = param -> Exwaldpdf(exp.(param)..., bin_centre, true) # renormalized Exwald
#     # KL_divergence = (param, grad) -> KLD(f_bin, dist(param), bw)
#     KL_divergence = (param, grad) -> sKLD(ISI, d->Exwaldpdf(exp.(param)..., d) )


#     #optStruc = Opt(:LN_NELDERMEAD,3)  # set up 3-parameter NLopt optimization problem

#     # set up optimization
#  #   optStruc = Opt(:LN_NELDERMEAD, length(pInit))  
#     # optStruc = Opt(:LN_PRAXIS, length(pInit)) 
#     # NLopt.min_objective!(optStruc, KL_divergence)      
#     # NLopt.lower_bounds!(optStruc, zeros(Float64, length(pInit)))  # constrain parameters > 0
#     # NLopt.xtol_abs!(optStruc, 1.0e-12)
#     f = OptimizationFunction(KL_divergence)
#     Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
#     sol = solve(Prob, NLopt.LN_NELDERMEAD(), reltol = 1.0e-9)
 
#   #  fitted_param = (sol.u[1], sol.u[1]/sol.u[2]^2, sol.u[3])

#     #@infiltrate
#     return tuple(exp.(sol.u)...) , sol.objective
# end

# fit Exwald distribution to interspike intervals
# returns parameters μ, λ, τ
# Fit_Exwald_to_ISI(ISI, [mu, lambda, tau],  [0.1, 0.1, 0.1])
function Fit_Exwald_to_ISI(ISI::Vector{Float64},
            pInit::Vector{Float64}=[NaN, NaN, NaN]; max_iter=100, tol=1e-6)
    
    # initial estimates from summary stats
    # if cv is small then μ ≈ mean(ISI), τ ≪ mean(ISI) and λ is estimated from IG approximation
    # if cv is large then τ ≈ mean(ISI), μ ≪ mean(ISI) and IG variance ≪ data variance ≈ τ² 
    if any(isnan.(pInit))  # if pInit was not specified or is invalid
        avg = mean(ISI)
        V = var(ISI) 
        cv = sqrt(V)/avg
        w = min(cv, 0.99)
        τ = w*avg
        μ = (1.0-w)*avg
        λ = μ/w^2
        pInit = [μ, λ, τ]
    end # otherwise pInit was passed as argument
  
    LB = [1.0e-4, 1.0e-3, 1.0e-6]  # lower bounds (empirical, from PP&H data)
    UB = [10.0*avg, 100.0*pInit[2], 10.0*avg ]  

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is Kullback-Leibler divergence from data to model
    KL_divergence = (param, grad) -> sKLD(ISI, d->Exwaldpdf(param..., d) )

    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_NELDERMEAD(), reltol = 1.0e-9)
 
    return tuple(sol.u...) , sol.objective
end

# fit Exwald model to spontaneous spike train generated by fractional SLIF model
# SLIF parameters are (μ, λ, τₘ) and q, where τₘ is mean-reverting time constant 
#                                      (= membrane time constant in LIF neuron)
# Exwald parameters are (μ, λ, τₓ).
# MGP March 2026
function fit_Exwald_to_qSLIF(Param::Tuple{Float64, Float64, Float64}, q::Float64 = 0.0, N::Int64=10000, 
                pInit::Vector{Float64}=[NaN, NaN, NaN], 
                meanrange::Tuple{Float64, Float64} = (0.0, 1.0), timeout::Float64=1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    # nb qSLIF parameters specified by limiting IG(mu, lambda) and membrane tau
    # SLIFparam gets the qSLIF neuron coeffs
    qSLIFneuron, SLIFparam = make_qSLIF_neuron(Param..., q, dt=dt) 

    ISI = interspike_intervals(qSLIFneuron, t->0.0, N) 

    if length(ISI)<N  || mean(ISI)<meanrange[1] || mean(ISI)>meanrange[2] 
    # neuron does not generate plausible intervals
        return (NaN, NaN, NaN), NaN   # don't even try 
    end

    fitted_param, objective = Fit_Exwald_to_ISI(ISI, pInit)

    return fitted_param, objective, ISI, SLIFparam

end


# fit qSLIF neuron model to Exwald distribution of spontaneous interspike intervals
# Target ISI distribution is Exwald(μ, λ, τ, t)
# N is number of intervals to generate for fitting to the target
# nb qSLIF neuron is parameterized using IG parameters μ & λ corresponding to limiting case
#    of large membrane time constant, in which case ISI distribution is IG.
# BUT this function returns fitted qSLIF model coeffs a, σ, τ, q
# e.g. call
function fit_qSLIF2Exwald(mu::Float64, lambda::Float64, tau::Float64, N::Int64=5000)
    
    pInit = [100.0, .2, 0.01, .0]
  
    LB = [1.0e-4, 1.0e-5, 0.005, 0.0]  # lower bounds 
    UB = [1000.0,    25.0,   0.05,  0.5 ]  

    grad = zeros(Float64, length(pInit))   # required input to fitting code but not used

    # Goodness of fit is Kullback-Leibler divergence 
    #   from qSLIF spontaneous ISI distribution to specified Exwald model
    KL_divergence = (param, grad) -> begin
        Random.seed!(42)
        qSLIF2Exwald_KLD(param,[mu, lambda, tau], N )
    end
    f = OptimizationFunction(KL_divergence)
    Prob = Optimization.OptimizationProblem(f, pInit, grad, lb=LB, ub = UB)
    sol = solve(Prob, NLopt.LN_SBPLX(), reltol = 1.0e-3, abstol = 1.0e-3, maxiters = 1000)
 
    return tuple(sol.u...) , sol.objective
end

# Fit a sine wave to spike train (spike times) at specified frequency Hz
# return amplitude and phase
function fit_sine2spiketrain_Fourier(spt::Vector{Float64}, f::Float64, T::Float64, 
            Ncycles::Int64=1, Nrep::Int64=1)
    n  = length(spt)
    φ  = 2π .* f .* spt
    A  =  2/T * sum(sin.(φ))    # sine coefficient
    B  =  2/T * sum(cos.(φ))    # cosine coefficient
    r0 = n / T                  # mean rate

    amplitude = sqrt(A^2 + B^2)/(Ncycles*Nrep)
    phase     = atan(-B, A)

    return (amplitude=amplitude, phase=phase, r0=r0, A=A, B=B)
end