# functions that are not needed any more. probably.


# map points on a circle around a point in SLIF parameter space 
# to Exwald parameters. Changed units from s to ms.
function ringmap_SLIF2Exwald(N::Int64=400)
 
    
    # SLIF tau-v0 parameter centre point (ms)
    tau_0, v0_0 = (25.0, 3000.0) 

    # SLIF current noise amplitude 
    sigma_s = 0.005

    # (log) radius of circle in SLIF parameter space
    R = .1

    # points on circle
    nn = 4
    tau_s, v0 = log_circle(x_c, y_c, R, nn)

    # Save/return fitted Exwald parameters and goodness of fit
    EXWparam = fill((NaN, NaN, NaN), nn+1)
    KLD = zeros(nn)



    # fit centre point
    EXWparam[1] = fit_Exwald_to_SLIF( (v0_0, sigma_s, tau_0), 0.1, N)
    println(1, ", ", EXWparam[1])

    for i in 2:(nn+1)
   # @infiltrate
     println(i, v0[i-1], tau_s[i-1])
        EXWparam[i] = fit_Exwald_to_SLIF( (v0[i-1], sigma_s, tau_s[i-1]), 0.1, N) 
        println(i, ", ", EXWparam[i])

    end 

    F = Figure()
    ax1 = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "SLIF τ",
                ylabel = "SLIF v0",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    xlims!(ax1, 1.0e-3, 1.0e1)
    ylims!(ax1, 1.0e0, 5.0e2)
    ax1.title = @sprintf("(%.4f, %.4f), L = %.1f, S = %.2f, sigma = %.4f",
                tau_0, v0_0, L, S, sigma_s)

    scatter!(ax1, tau_s, v0)

     ax2 = Axis(F[1,2], xscale = log10, yscale = log10, 
                xlabel = "Exwald τ",
                ylabel = "Exwald λ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
     xlims!(ax2, 1.0e-5, 1.0e-1)
     ylims!(ax2, 1.0e-2, 1.0e2)   

  scatter!(ax2, [EXWparam[i][3] for i in 1:n_pts], 
                [EXWparam[i][2] for i in 1:n_pts])


    display(F)   

   # uncomment next line (& maybe pick a different file name) to auto-save 
   # using JLD2
   #jldsave("OU2EXW_line_19Jan26D.jld2"; EXWparam, grid)
   # DATA = load("filename") to recover

    return    EXWparam, tau_s, v0, F, ax1, ax2

end



# map points along a line in SLIF parameter space 
# to Exwald parameters
function linemap_SLIF2Exwald(N::Int64=400)
 
    # using map_SLIF2Exwald (above) identified a line in 
    # (v0, sigma, tau_s) SLIF parameter space corresponding (roughly)
    # to the 1st principle axis of Exwald models (PP&H figure 3).
    # This line goes through a = (.00774, .005, 2.385) and b = (.464, .005, 50.0) 
    # in SLIF space.  Sigma (noise amplitude) is constan so we drop to 2D. 
    # The line has slope S = 0.634 in log-log axes, 
    # i.e. in "real" parameter space there is a power law relatioship
    # between v0 and tau_s:  v0 = 10^v00 * tau_s^S.
    # The distance between endpoints is about L = 2 (log units).
    # So a rough starting point, identified by a coarse grid search,
    # is that the SLIF parameters are on a line of length L with slope S
    # starting at a.
    
    # SLIF parameter line parameters
    tau_0, v0_0 = (.005, 2.4) #(.003, 0.6) #(.00774, 2.385) 
    L =  3.0
    S =  0.67
    sigma_s = 0.005
    n_pts = 32

    # get n points along this line
    tau_s, v0 = points_along_lineLogLog(tau_0, v0_0, L, S, n_pts)

    # F = Figure()
    # ax = Axis(F[1,1], xscale = log10, yscale = log10, 
    #             xlabel = "SLIF τ",
    #             ylabel = "SLIF v0",
    #             xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    # xlims!(ax, 1.0e-3, 1.0e1)
    # ylims!(ax, 1.0e0, 1.0e2)

    # Save/return fitted Exwald parameters and goodness of fit
    EXWparam = fill((NaN, NaN, NaN), n_pts)
    KLD = zeros(n_pts)

    #@infiltrate

    for i in 1:n_pts

        EXWparam[i] = fit_Exwald_to_SLIF( (v0[i], sigma_s, tau_s[i]), 
                    0.1, N) #, (0.0005, 0.1))
        println(i, ", ", EXWparam[i])

    end 

    F = Figure()
    ax1 = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "SLIF τ",
                ylabel = "SLIF v0",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    xlims!(ax1, 1.0e-3, 1.0e1)
    ylims!(ax1, 1.0e0, 5.0e2)
    ax1.title = @sprintf("(%.4f, %.4f), L = %.1f, S = %.2f, sigma = %.4f",
                tau_0, v0_0, L, S, sigma_s)

    scatter!(ax1, tau_s, v0)

     ax2 = Axis(F[1,2], xscale = log10, yscale = log10, 
                xlabel = "Exwald τ",
                ylabel = "Exwald λ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
     xlims!(ax2, 1.0e-5, 1.0e-1)
     ylims!(ax2, 1.0e-2, 1.0e2)   

  scatter!(ax2, [EXWparam[i][3] for i in 1:n_pts], 
                [EXWparam[i][2] for i in 1:n_pts])


    display(F)   

   # uncomment next line (& maybe pick a different file name) to auto-save 
   # using JLD2
   #jldsave("OU2EXW_line_19Jan26D.jld2"; EXWparam, grid)
   # DATA = load("filename") to recover

    return    EXWparam, tau_s, v0, F, ax1, ax2

end


# construct a map from fractional Ornstein-Uhlenbeck FPT (SLIF neuron)
# parameter space to Exwald parameters over a grid of points
# This is very slow, used to explore SLIF parameter space & identify
# a region where ISI distributions of fractional SLIF models match
# empirical ISI distributions of vestibular afferents  
function map_SLIF2Exwald(N::Int64=8000)
 
    # default ..., 41))[1:40]
    # 11))[1:10]
    # 21))[1:20]   
    # 13))[1:12]

    # following grid search maps out the region of (v0, sigma, tau) 
    # parameter space of the fractional SLIF model 
    # corresponding to empirical Exwald ISI models
    # (i.e. roughly identifies an Orstein-Uhlenbeck SDE model with 
    #  fractional noise input, as a model of canal afferents)
    v0 = collect(logrange(1.0, 50.0, 10))
    N_v0 = length(v0)
    sigma = [.004, .005, .006]  #collect(logrange(5.0e-4, 5.0e-2, 3)) #collect(logrange(1.0e-4, 1.0e4, 10)) #[1:6]    # from e2-e6
    N_sigma = length(sigma)
    taus = collect(logrange(1.0e-3, 1.0e1, 10)) #[1:10]
    N_tau = length(taus)

    colour = 0.01

    F = Figure()
    ax = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "LIF time constant τ",
                ylabel = "Exwald τ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
   # ax.xticks = [.005, .01, .02, .05]
  #  ax.title = @sprintf "LIF μ= %.4f" v0
    xlims!(ax, .0005, .1)
    ylims!(ax, .000001, 0.5)

    # Save/return fitted Exwald parameters and goodness of fit
    EXWparam = fill((NaN, NaN, NaN), (N_v0, N_sigma, N_tau))
    KLD = zeros(N_v0, N_sigma, N_tau)

    #@infiltrate

   # for i in 1:N_mu
    for i in 1:N_v0
        for j in 1:N_sigma
            pInit = (NaN, NaN, NaN)
            for k in 1:N_tau
                #v0 = 1.0/(1.0 - exp(-mu_0[i]/taus[k]))  # drift required for expected FPT = mu[i]
                println(i, ", ", j, ", ", k)
                EXWparam[i, j, k] = fit_Exwald_to_SLIF( (v0[i], sigma[j], taus[k]), taus[k],
                                                         N, pInit) #, (0.0005, 0.1))
                println(EXWparam[i, j, k])
               # pInit = EXWparam[i, j, k]
            end
            lines!(taus, [EXWparam[i,j,n][3] for n in 1:N_tau])
       
        end
        #labl = @sprintf "%.4f" sigma[j]
           display(F)   


    end

 #   axislegend(ax, position = :rt) # :lb means 'left bottom'
   # display(F)
   grid = (v0, sigma, taus)

   # uncomment next line (& maybe pick a different file name) to auto-save 
   # using JLD2
   jldsave("OU2EXW_19Jan26D.jld2"; EXWparam, grid)
   # DATA = load("filename") to recover

    return    EXWparam, grid , F

end



# construct a map from fractional Ornstein-Uhlenbeck FPT (SLIF neuron)
# parameter space to Exwald parameters over a grid of points
# This is very slow, used to explore SLIF parameter space & identify
# a region where ISI distributions of fractional SLIF models match
# empirical ISI distributions of vestibular afferents  
function map_SLIF2Exwald(N::Int64=8000)
 
    # default ..., 41))[1:40]
    # 11))[1:10]
    # 21))[1:20]   
    # 13))[1:12]

    # following grid search maps out the region of (v0, sigma, tau) 
    # parameter space of the fractional SLIF model 
    # corresponding to empirical Exwald ISI models
    # (i.e. roughly identifies an Orstein-Uhlenbeck SDE model with 
    #  fractional noise input, as a model of canal afferents)
    v0 = collect(logrange(1.0, 50.0, 10))
    N_v0 = length(v0)
    sigma = [.004, .005, .006]  #collect(logrange(5.0e-4, 5.0e-2, 3)) #collect(logrange(1.0e-4, 1.0e4, 10)) #[1:6]    # from e2-e6
    N_sigma = length(sigma)
    taus = collect(logrange(1.0e-3, 1.0e1, 10)) #[1:10]
    N_tau = length(taus)

    colour = 0.01

    F = Figure()
    ax = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "LIF time constant τ",
                ylabel = "Exwald τ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
   # ax.xticks = [.005, .01, .02, .05]
  #  ax.title = @sprintf "LIF μ= %.4f" v0
    xlims!(ax, .0005, .1)
    ylims!(ax, .000001, 0.5)

    # Save/return fitted Exwald parameters and goodness of fit
    EXWparam = fill((NaN, NaN, NaN), (N_v0, N_sigma, N_tau))
    KLD = zeros(N_v0, N_sigma, N_tau)

    #@infiltrate

   # for i in 1:N_mu
    for i in 1:N_v0
        for j in 1:N_sigma
            pInit = (NaN, NaN, NaN)
            for k in 1:N_tau
                #v0 = 1.0/(1.0 - exp(-mu_0[i]/taus[k]))  # drift required for expected FPT = mu[i]
                println(i, ", ", j, ", ", k)
                EXWparam[i, j, k] = fit_Exwald_to_SLIF( (v0[i], sigma[j], taus[k]), taus[k],
                                                         N, pInit) #, (0.0005, 0.1))
                println(EXWparam[i, j, k])
               # pInit = EXWparam[i, j, k]
            end
            lines!(taus, [EXWparam[i,j,n][3] for n in 1:N_tau])
       
        end
        #labl = @sprintf "%.4f" sigma[j]
           display(F)   


    end

 #   axislegend(ax, position = :rt) # :lb means 'left bottom'
   # display(F)
   grid = (v0, sigma, taus)

   # uncomment next line (& maybe pick a different file name) to auto-save 
   # using JLD2
   jldsave("OU2EXW_19Jan26D.jld2"; EXWparam, grid)
   # DATA = load("filename") to recover

    return    EXWparam, grid , F

end



# 3D plot of SLIF parameter as function of 
# fitted Exwald parameters, generated by map_SLIF2Exwald()
# sp is SLIF parameter (to be represented by color at plotted points)
#    valid values are "mu", "lambda" and "tau"
function plot3D_fittedExwald_vs_SLIF(filename::String, sp::String)

    D = load(filename)

    # EXWparam is 3D array 
    # whose entries are tuples of Exwald parameters
    # fitted to SLIF neuron ISI data.
    EXWparam = D["EXWparam"]

    # vex specifies points where there was an attempt
    # to fit an Exwald model to SLIF neuron data.
    # This fails at most points in the grid, where the 
    # corresponding entry in EXWparam is (NaN, NaN, NaN) 
    vex = D["grid"]
    sMu = vex[1]      # SLIF mu
    Nm = length(sMu)
    sSigma = vex[2]  # SLIF lambda
    Nl = length(sSigma)
    sTau = vex[3]     # SLIF tau
    Nt = length(sTau)

   # DKL = D["V"]

    # index of SLIF parameter to encode in point color
    w = findfirst(==(sp), ("mu", "sigma", "tau") )

    # find points in Exwald parameter space
    # where an Exwald model was fitted to a SLIF model 
    xMu = []
    xLambda = []
    xTau = []

    # filtered SLIF parameters
    fsMu = []
    fsSigma = []
    fsTau = []

    # S = []      # for SLIF parameters
    pointColor = []
    cc = 0
    for i in 1:Nm
        for j in 1:Nl
            for k in 1:Nt
                
                cc += 1

                if !isnan(EXWparam[i,j,k][1])

                    # if sTau[k]>.005 && sTau[k]<.04 &&
                    #    sMu[i]<.01 && sMu[i]>.008 && sSigma[j]<5. &&
                    #     (EXWparam[i,j,k][1]+EXWparam[i,j,k][3]) > .01 &&
                    #      (EXWparam[i,j,k][1]+EXWparam[i,j,k][3]) < .1

                    push!(xMu, EXWparam[i,j,k][1])
                    push!(xLambda, EXWparam[i,j,k][2])                    
                    push!(xTau, EXWparam[i,j,k][3]) 

                    push!(fsMu, sMu[i])
                    push!(fsSigma, sSigma[j])
                    push!(fsTau, sTau[k])

                    #S, _, _ = Exwald_entropy(EXWparam[i,j,k])
                    push!(pointColor, [sMu[i], sSigma[j], sTau[k]][w])
                    
                    push!(pointColor, [sMu[i], sSigma[j], sTau[k]][w])
                    #push!(pointColor, DKL[i,j,k])
                    # push!(pointColor, sTau[k])

                    # end
                end
            end
        end
    end

       
    # Create figure
    F = Figure(size=(1200, 600))
    axX = Axis3(F[1, 1],
               xlabel="TAU",
               ylabel="LAMBDA",
               zlabel="MU")
     xlims!(axX, (-5,-1))
     ylims!(axX, (-2, 2))
     zlims!(axX, (-3, 0)) 
              # aspect=axis_equal ? :data : :auto)

           #   @infiltrate
    
    # Scatter plot
   scatter!(axX, log10.(xTau), log10.(xLambda), log10.(xMu), 
             color=pointColor)

    axS = Axis3(F[1, 2],
               xlabel="TAU",
               ylabel="SIGMA",
               zlabel="MU")
   #            title=title)
    #   xlims!(axS, (-4,0))
    #   ylims!(axS, (-4, 0))
    #   zlims!(axS, (-2, 2))
    scatter!(axS, log10.(fsTau), log10.(fsSigma), log10.(fsMu),
              color=pointColor)
            #  colormap=colormap,
            #  colorrange=colorrange,
            #  markersize=markersize,
            #  alpha=alpha)
    
    #Colorbar
    # if show_colorbar
    #     Colorbar(fig[1, 2],
    #              limits=colorrange,
    #              colormap=colormap,
    #              label="Value")
    # end

    display(F)

    # return the plotted points as Nx3 matrices
    return hcat((xTau, xLambda, xMu)...), hcat((fsTau, fsSigma, fsMu)...)

end


# Compare ISI distribution of SLIF neuron
# with Exwald distribution used to specify its parameters 
function demo_SLIFfromExwald(EXWparam::Tuple{Float64, Float64, Float64},
            N::Int64)

    # load OU2EXW map
    DATA=load("OU2EXW_40x40x40_4000_1.jld2");
    EXWmap = DATA["EXWparam"]
    grid = DATA["vex"]

        
    # transform Exwald parameters to SLIF parameters
    OUparam = Exwald2SLIF(EXWparam, grid, EXWmap)

    fittedEXWparam, ob = fit_Exwald_to_OU(OUparam, N)

    # construct SLIF neuron
    SLIFneuron = make_OU_neuron(OUparam)[1]

    # N spontaneous intervals
    ISI = interspike_intervals(SLIFneuron, t->0.0, N)

   # @infiltrate

    # frequency histogram
    T = 1.25*maximum(ISI)  # longest interval 
    bw = T/128; #max(Int(round(N/500)), 32)   # at least 32 bins
    bin_edges = 0.0:bw:T
    t = collect(0.0:(bw/10):T)
    H = fit(Histogram, ISI, bin_edges) 

    # relative frequency histogram 
    f_bin = H.weights./(sum(H.weights)*bw)
    bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)

    F = Figure()
    ax = Axis(F[1,1])

    stairs!(bin_centres, f_bin)
    grid = collect( 0.0:bw/10.0:T )
    lines!(grid, Exwaldpdf(EXWparam..., grid), linewidth=2)
    lines!(grid, Exwaldpdf(fittedEXWparam..., grid), linewidth=2)

    display(F)

    return OUparam, fittedEXWparam, F

end




# OU tau as function of Exwald parameters mu, lambda, tau
# res = map resolution
function plot_SLIF_tau_vs_Exwald_params(res::Int64)

    # load OU2EXW map
    DATA=load("OU2EXW_40x40x40_4000_1.jld2");
    EXWparam = DATA["EXWparam"]
    grid = DATA["vex"]

    # μ, τ and λ ranges from PP&H paper 
    mu = collect(logrange(.001, .1, length = res))
    M = length(mu)
    lambda = collect(logrange(.01, 100.0, length = res)) 
    L = length(lambda)
    tau = collect(logrange(.00001, .1, length = res))
    T = length(tau)

    OUtau = NaN*zeros(M,L,T)
    println("")
    for i in 1:M 
        for j in 1:L
            for k in 1:T
                OUtau[i,j, k] = Exwald2SLIF((mu[i],lambda[j], tau[k] ),
                        grid,
                        EXWparam)[3]
            end
            print("(", i,",", j, "), ")
        end
        println("")
    end

    # F = Figure()
    # ax = Axis(F[1,1], xscale = log10, yscale = log10,
    #  xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    # heatmap!(tau, lambda, OUtau, colorrange = (.005, .025))

    # display(F)

    return OUtau, mu, lambda, tau

end


# OU tau as function of Exwald tau and lambda for given Exwald mu 
# res = map resolution
function plot_SLIF_tau_given_Exwald_tau_lambda(mu::Float64, res::Int64)

    # load OU2EXW map
    DATA=load("OU2EXW_40x40x40_4000_1.jld2");
    EXWparam = DATA["EXWparam"]
    grid = DATA["vex"]

    # τ and λ ranges from PP&H paper 
    tau = collect(logrange(.00001, .1, length = res))
    M = length(tau)
    lambda = collect(logrange(.01, 100.0, length = res)) 
    N = length(lambda)

    OUtau = zeros(M,N)
    println("")
    for i in 1:M 
        for j in 1:N 
            OUtau[i,j] = Exwald2SLIF((mu,lambda[j], tau[i] ),
                        grid,
                        EXWparam)[3]
        end
        print(i," ")
    end
    println("")

    F = Figure()
    ax = Axis(F[1,1], xscale = log10, yscale = log10,
     xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    heatmap!(tau, lambda, OUtau, colorrange = (.005, .025))

    display(F)

    return OUtau, tau, lambda

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


# closure to generate pink noise step dP
# by adding N white noise samples (Voss pink noise generator)
function make_pink_noise(N=24)

    count = 0
    whites = randn(N)   # samples of white noise
    P = 0.0

    return function()

        count = count + 1
        for j in 0:(N-1)
            if count % (2^j) == 0
                whites[j+1] = randn()[]
            end
        end

        return mean(whites)
    end

end

# closure to generate pink noise sample dW
# by adding N independent samples from heavy-tail LambertW-Normal distribution
# nb lambertWNormal_sample(0.0) is standard Gaussian
function make_pinkish_noise(d::Float64=.1, N::Int64=24)

    count = 0
    whites = [lambertWNormal_sample(d) for i in 1:N]   # samples of white noise


    return function(reset::Bool=false)

        if reset  # reset after each spike to get independent intervals
            count = 0
            whites = [lambertWNormal_sample(d) for i in 1:N]   # samples of white noise
        end

            count = count + 1
            for j in 0:(N-1)
                if count % (2^j) == 0
                    whites[j+1] = lambertWNormal_sample(d)
                end
            end

        return mean(whites)

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
# It picks the threshold required to get the requested mean rate.
# sample size = length(interval)
function Exponential_sample(interval::Vector{Float64}, tau::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    m = 0.0
    s = 1.0
    threshold = TriggerThreshold_from_PoissonTau(m, s, tau, dt)
    # @infiltrate
    ThresholdTrigger_simulate(interval, m, s, threshold, dt)
    (m, s, threshold)  # return trigger mechanism parameters
end



# Intervals from inhomogenous Exponential distribution by noisy integrate-and-fire
# Difference from previous version is here you specify the required rate modulation
# instead of just the modulating function - it calibrates the input to vary the spike rate by
#  a specified amount.
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




# First passage times of drift-diffusion to barrier (integrate noise to threshold)
# generates samples from Wald aka Inverse Gaussian distribution, 
# specified by coefficients of the drift-diffusion process.
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
# ie computes drift-diffusion coeffs from Wald parameters
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # drift-diffusion model parameters from Wald parameters
    (v, s, a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    println(v, " ", s, " ", a)
    FirstPassageTime_simulate(interval, v, s, a, dt)
    return interval
end

# intervals from inhomogenous Poisson process
# by (time-varying) integrate-and-fire,
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
#
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

# function make_fractional_SLIF_neuron(
#     SLIF_param::Tuple{Float64, Float64, Float64}, q::Float64, 
#     x0::Float64=0.0; 
#     dt::Float64=DEFAULT_SIMULATION_DT, f0::Float64=1e-2, f1::Float64=2e1)

#     #@infiltrate

#     # Fractional SLIF neuron
#         # extract OU parameters
#     (mu, lambda, tau) = SLIF_param

#     # First passage time model parameters for τ = 0.0 (Inverse Gaussian/Wald model)
#     # with barrier height = 1.0
#     (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda, "barrier", 1.0)
 

#     # input gain (how much the drift rate is affected by input)
#     G = 1.0

#     # convert frequency band from Hz to rad/s
#     wb = 2.0*pi*f0
#     wh = 2.0*pi*f1

#     # Approximation of order 2N+1 (so N=2 is 5th order)
#     N = 5
    
#     # Compute Oustaloup parameters
#     K, poles, xeros = oustaloup_zeros_poles(q, N, wb, wh)
    
#     # Compute residues and pole dynamics coefficients (the p_i = ω_k >0 for v' = -p_i v + y)
#     residues, _ = oustaloup_residues(K, poles, xeros)
#     p_i = poles  # p_i = ω_k for the dynamics v' = -p_i v + y
    
#     M = length(poles)  # ... = 2N+1

#     # Augmented state: x = [y, v1, ..., vM]
#     x = [x0; zeros(M)]
#     du = zeros(length(x))

#     Threshold = 1.0
#    # Random.seed!(4242)

#     # neuron update function given u(t) 
#     function qSLIF(u::Function, t::Float64)

#      #   @infiltrate

#      #   dx = (-x/tau + v0 + G*u(t))*dt + s * randn(1)[] * sqrt(dt) 
 

#         ut = v0 + G*u(t) + s*randn(1)[]/sqrt(dt)   # input at t

#         vs = @view x[2:end]

#        # @infiltrate
        
#         # Approximate D^q u 
#         if (q==0.0) 
#             approx_dq = ut
#         else
#             approx_dq = K * ut
#             for i in 1:M
#                 approx_dq += residues[i] * vs[i]
#             end
#         end
        
#         # state update
#         du[1] =  approx_dq - x[1]/tau
        
#         # Auxiliary state update
#         for i in 1:M
#             du[1 + i] = -p_i[i] * vs[i] + ut
#         end

#      #   @infiltrate

#         # Euler integration
#         for i in 1:length(x)
#             x[i] += du[i] * dt
#         end

#         if x[1] >= Threshold      
#             x[1] -= Threshold 
#            # vs .= 0.0
#             return true
#         else
#             return false  
#         end
    
#     end

#     # return closure
#     return qSLIF 
# end


"""
Load all .jld2 files and return as vector of named tuples
Each tuple contains filename and data
"""
function process_OU2EXW(folder_path::String)

    # get names of all .jld2 files in folder
    filename = filter(f -> endswith(f, ".jld2"), readdir(folder_path))

    # iniitialize using data from first file
    RawData = load(joinpath(folder_path, filename[1]))
    EXWparam = RawData["EXWparam"]
    Goodness = RawData["Goodness"]
    M, N, P  = size(EXWparam)


    # filter using the remaining files
    for f in 2:length(filename)

        RawData = load(joinpath(folder_path, filename[f]))  

        for i in 1:M 
            for j in 1:N 
                for k in 1:P 
                    if RawData["EXWparam"][i,j,k][3] > EXWparam[i,j,k][3]

                        EXWparam[i,j,k] = RawData["EXWparam"][i,j,k]
                        Goodness[i,j,k] = RawData["Goodness"][i,j,k]

                    end
                end
            end
        end
    end

    # shoulda saved these with output data ... 
    # check the following in demo_OU2Exwald(...)
    mu_o = vec([.001 .002 .005 .01 .02 .05])
    N_mu = length(mu_o)    # = N 
    N_lambda = 8           # = M 
    lambda_o = collect(logrange(1.0e-2, 5.0e1, length = N_lambda))
    N_tau = 32   # = P 
    tau_o = collect(logrange(1.0e-3, 5.0e-2; length=N_tau))

    F = Figure()
    ax = Axis(F[1,1], xscale = log10, yscale = log10, 
                xlabel = "LIF μ",
                ylabel = "Exwald μ",
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
   # ax.xticks = [.005, .01, .02, .05]
  #  ax.title = @sprintf "LIF μ= %.4f" mu_o
    # xlims!(ax, .001, .1)
    # ylims!(ax, .000001, 0.1)

    
    for i in 1:N 
        for j in 1:P 
            lines!(mu_o, [EXWparam[n,i,j][1] for n in 1:M]) 
        end
        #labl = @sprintf "%.4f" lambda_o[j]
  
    end
            display(F)   

    return EXWparam, Goodness, F
end

# linear interpolate y(xx)
# given y(x1) = y1, y(x2) = y2 and x1 < xx < x2
# if x1,x2, y1 or y2 is NaN then return NaN
# unless xx is approximately x1 or x2
function linterp(x::Vector{Float64}, y::Vector{Float64}, xx::Float64)

    if isnan(x[1])
        if isapprox(x[2], xx, rtol = .01)  # nb isapprox(NaN, NaN, ...) is always false
            return y[2]  # possibly NaN
        else 
            return NaN
        end
    elseif isnan(x[2])
        if isapprox(x[1], xx, rtol = .01)  # nb isapprox(NaN, NaN, ...) is always false
            return y[1]  # possibly NaN
        else
            return NaN
        end        
    elseif xx > x[1] && xx < x[2]
        return y[1]*(x[2]-xx)/(x[2]-x[1]) + y[2]*(xx-x[1])/(x[2] - x[1])
    else
        error("Interpolation failed (xx must be between x1 and x2)")
    end

end


# Trilinear interpolation in unit cube
function trilinear_normalized(xd::Real, yd::Real, zd::Real, vals::Array{Float64, 3})
    v000 = vals[1, 1, 1]
    v100 = vals[2, 1, 1]
    v010 = vals[1, 2, 1]
    v110 = vals[2, 2, 1]
    v001 = vals[1, 1, 2]
    v101 = vals[2, 1, 2]
    v011 = vals[1, 2, 2]
    v111 = vals[2, 2, 2]

    return (v000 * (1 - xd) * (1 - yd) * (1 - zd) +
            v100 * xd * (1 - yd) * (1 - zd) +
            v010 * (1 - xd) * yd * (1 - zd) +
            v110 * xd * yd * (1 - zd) +
            v001 * (1 - xd) * (1 - yd) * zd +
            v101 * xd * (1 - yd) * zd +
            v011 * (1 - xd) * yd * zd +
            v111 * xd * yd * zd)
end

# Bilinear interpolation in unit square
function bilinear_normalized(u::Real, v::Real, vals::Matrix{Float64})
    return ((1 - u) * (1 - v) * vals[1, 1] +
            u * (1 - v) * vals[2, 1] +
            (1 - u) * v * vals[1, 2] +
            u * v * vals[2, 2])
end

# project 3D array A3 to 2D array by averaging over dimension d
# ignore NaNs in the average (unless everything is NaN, then average = NaN) 
function projectmap(map3D::Array, dim::Int) 
    
    N  = collect(size(map3D))                     
    D  = findall(d-> d!=dim, [1, 2, 3])    # dimensions to keep
    d1 = D[1] 
    d2 = D[2]
    map2D = zeros(N[d1],N[d2])    # blank screen for 2D projection along dim

    for i in 1:N[d1] 
        for j in 1:N[d2]

            count = 0     
            for k in 1:N[dim]

                # permute indices to average over dim
                p = (i,j,k)  # if dim==3
                if dim==1
                    p = (j,k,i)
                elseif dim==2
                    p = (i,k,j)
                end

                if !isnan(map3D[p...])
                    map2D[i,j] += map3D[p...]
                    count += 1
                end
            end

            if count==0   # all NaNs 
                map2D[i,j] = NaN
            else
                map2D[i,j] /= count
            end

        end
    end

    return map2D

end


# 
# e.g. map: jldsave("OUtau_mu_lambda_tau_Cloud.jld2"; OUtau, mu, lambda, tau)  
function plot_3D_map_as_cloud(data::Array{Float64, 3},
                               x_vals, y_vals, z_vals;
                               colormap=:viridis,
                               markersize=5,
                               colorrange=nothing,
                               alpha=1.0,
                               threshold=nothing,  # Only plot values above threshold
                               value_range=nothing,  # Only plot values in (min, max)
                               show_colorbar=true,
                               xlabel="x",
                               ylabel="y", 
                               zlabel="z",
                               title="3D Point Cloud",
                               axis_equal=false)
    
    x_vec = collect(x_vals)
    y_vec = collect(y_vals)
    z_vec = collect(z_vals)
    
    @assert length(x_vec) == size(data, 1) "X values must match first dimension"
    @assert length(y_vec) == size(data, 2) "Y values must match second dimension"
    @assert length(z_vec) == size(data, 3) "Z values must match third dimension"
    
    # Extract valid points
    points_x = Float64[]
    points_y = Float64[]
    points_z = Float64[]
    colors = Float64[]
    
    nx, ny, nz = size(data)
    
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                val = data[i, j, k]
                
                # Skip NaN
                if isnan(val)
                    continue
                end

                # # skip short or long
                # mean_interval = x_vec[i]+z_vec[k]
                # if mean_interval < 0.05 || mean_interval > 0.1
                #     continue
                # end 
                
                # Apply threshold
                if threshold !== nothing && val < threshold
                    continue
                end
                
                # Apply value range
                if value_range !== nothing
                    vmin, vmax = value_range
                    if val < vmin || val > vmax
                        continue
                    end
                end
                
                push!(points_x, x_vec[i])
                push!(points_y, y_vec[j])
                push!(points_z, z_vec[k])
                push!(colors, val)
            end
        end
    end
    
    if isempty(colors)
        @warn "No valid points to plot!"
        return nothing
    end
    
    # Color range
    if colorrange === nothing
        colorrange = (minimum(colors), maximum(colors))
    end
    
    # Create figure
    fig = Figure(size=(600, 600))
    ax3 = Axis3(fig[1, 1], 
               xlabel=xlabel,
               ylabel=ylabel,
               zlabel=zlabel,
               title=title)
    xlims!(ax3, (-4,0))
    ylims!(ax3, (-2, 2))
    zlims!(ax3, (-5, -1))
              # aspect=axis_equal ? :data : :auto)
    
    # Scatter plot
    scatter!(ax3, log10.(points_x), log10.(points_y), log10.(points_z),
             color=colors,
             colormap=colormap,
             colorrange=colorrange,
             markersize=markersize,
             alpha=alpha)
    
    Colorbar
    if show_colorbar
        Colorbar(fig[1, 2],
                 limits=colorrange,
                 colormap=colormap,
                 label="Value")
    end


    # ax_tau_mu = Axis(fig[1,2])
    # heatmap!(ax_tau_mu, projectmap(data, 2) )

    # ax_lambda_mu = Axis(fig[2,1])
    # heatmap!(ax_lambda_mu, projectmap(data, 3) )

    # ax_tau_lambda = Axis(fig[2,2])
    # heatmap!(ax_tau_lambda, projectmap(data, 1) )

    display(fig)


    println("Plotted $(length(colors)) points")
    
    return fig
end


function plot_linked_points(A::Matrix{Float64}, B::Matrix{Float64})
    # Basic validation
    size(A) == size(B) || error("Matrices A and B must have the same dimensions")
    
    # Create the figure
    fig = Figure(size = (1200, 600))
    
    # Define axes
    ax1 = Axis3(fig[1, 1], title = "Set A")
    ax2 = Axis3(fig[1, 2], title = "Set B")
    
    # This Observable tracks the index of the currently selected point
    # We initialize it to 0 (no selection)
    selected_idx = Observable(0)
    
    # Create color arrays that update when selected_idx changes
           c = fill(:blue, size(A, 1))
    colors = lift(selected_idx) do idx
      #  c = fill(:blue, size(A, 1))
        if idx > 0 && idx <= length(c)
            c[idx] = :red # Highlight color
        end
        return c
    end

    # Plot the points
    # We use rows as points: A[:, 1], A[:, 2], A[:, 3]
    markersize = 8
    plt1 = scatter!(ax1, log10.(A[:, 1]), log10.(A[:, 2]), log10.(A[:, 3]), color = colors, markersize = markersize)
    xlims!(ax1, -5, -1)
    ylims!(ax1, -2,2)
    zlims!(ax1, -4, 0)
    plt2 = scatter!(ax2, log10.(B[:, 1]), log10.(B[:, 2]), log10.(B[:, 3]), color = colors, markersize = markersize)

    # Interaction logic
    on(events(fig).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            # Pick the plot object under the mouse
            plt, idx = pick(fig)
            
            # Check if we clicked a point in plot 1 or plot 2
            if plt in (plt1, plt2) && idx > 0
                selected_idx[] = idx
                
                # Print coordinates to REPL
                println("\nSelected Point Index: $idx")
                println("Coord A: $(A[idx, :])")
                println("Coord B: $(B[idx, :])")
            end
        end
    end

    return fig, ax1, ax2
end


# set of n points uniformly distributed in log-log axes
# along a line between a = (x0, y0) and b = (x1,y1)
# (where a and b are untransformed coords) 
function points_along_lineLogLog(a, b, n)

    # transform to log-log space
    x0L, y0L = log10(a[1]), log10(a[2])
    x1L, y1L = log10(b[1]), log10(b[2])

    # slope in log-log space
    s = (y1L-y0L)/(x1L - x0L)
    println("Slope = ", s)

    # n points along the line
    xL = range(x0L, x1L, length = n)
    yL = range(y0L, y1L, length = n)

    # back transform 
    x = 10.0 .^ xL
    y = 10.0 .^ yL

    return x,y, s

end

# n equally spaced points along a straight line of length Length
# and slope slope in log-log axes
function points_along_lineLogLog(x0, y0, len, slope, n)
   
    # start point in log space
    x0L, y0L = log10(x0), log10(y0)
    
    # slope in radians
    θ = atan(slope)
    
    # distances along the line
    d = range(0, len, length=n)
    
    # log-space coordinates of points on line
    xL = x0L .+ d .* cos(θ)
    yL = y0L .+ d .* sin(θ)

    # Transform back 
    return 10 .^ xL, 10 .^ yL
end

using GLMakie


#  n points on a circle of radius 'r' in log-log axes
function log_circle(x_center, y_center, r, n=8)
    # 1. Move the center to log-space
    lc_x = log10(x_center)
    lc_y = log10(y_center)
    
    # 2. Generate angles from 0 to 2π
    #θ = range(0, 2π, length=n)
    θ = 2pi*(0:n-1)/n

    # 3. Calculate points on the circle in log-space
    # x = center + r*cos(θ), y = center + r*sin(θ)
    lx_points = lc_x .+ r .* cos.(θ)
    ly_points = lc_y .+ r .* sin.(θ)
    
    # 4. Transform back to original units
    return 10 .^ lx_points, 10 .^ ly_points
end

# # --- Visualization ---
# x_c, y_c = 100.0, 100.0  # Center of the circle
# radius = 0.5            # Radius in log-units (half an order of magnitude)

# x_vals, y_vals = log_circle(x_c, y_c, radius, 16)

# fig = Figure()
# ax = Axis(fig[1, 1], 
#     xscale = log10, 
#     yscale = log10,
#     aspect = DataAspect(), # Essential: makes 1 unit on x equal 1 unit on y visually
#     title = "Circle in Log-Log Space"
# )

# scatter!(ax, x_vals, y_vals, color = :magenta)
# scatter!(ax, [x_c], [y_c], color = :black) # Plot the center point

# display(fig)


# Uniform Gaussian rate filter
# Firing rate estimation by sampling Gaussians centred at spike times
#   spiketimes, sd of Gaussian filter, dt = sampling interval, T = end time
# MGP July 2024
# spiketime in seconds
# sd of Gaussian kernal for each spike (ifP length(sd)==1 then same kernel for all spikes)
# dt = sample interval in seconds
# pad = pad the start and end of the spike train with mirror image of spikes in first and last pad seconds 
#       (kluge to prevent edge effects. the padded ends are removed before return) (default no pads).
# last sample time (default rounded up to nearest second)
function GLR(spiketime::Vector{Float64}, sd::Vector{Float64}, dt::Float64, pad::Float64=0.0, T::Float64=maximum(spiketime))

    if length(sd) == 1
        sd = sd[] * ones(length(spiketime))
    end

    if length(sd) != length(spiketime)
        print("length mismatch: length of sd must equal number of spikes")
        return
    end


    # print("length(spiketime) = "), println(length(spiketime))
    # print("pad= "), println(pad)

    if pad > 0.0
        ifront = findall(spiketime[:] .< pad)[end:-1:1]  # indices of front pad spikes in reverse order (mirror) 
        iback = findall(spiketime[:] .> (T - pad))[end:-1:1]
        #infiltrate
        spiketime = pad .+ vcat(2.0 * spiketime[1] .- spiketime[ifront[1:end-1]], spiketime, 2.0 * spiketime[iback[1]] .- spiketime[iback[2:end]])
        sd = vcat(sd[ifront[2:end]], sd, sd[iback[2:end]])
    end
    # print("length(spiketime) = "), println(length(spiketime))

    # vector to hold result
    padN = Int(ceil(pad / dt))    # pad lengths
    sigN = Int(ceil(T / dt))      # signal length (number of sample points)
    N = sigN + 2 * padN           # padded signal length
    t = (1:N) * dt                  # time vector  (for return)
    r = zeros(N)                  # rate vector


    for i in 1:length(spiketime)
        k = Int(round(spiketime[i] / dt)) # ith spike occurs at this sample point
        n = Int(round(4.0 * sd[i] / dt))  # number of sample points within 4 sd each side of spike
        kernel = pdf(Normal(spiketime[i], sd[i]), k * dt .+ (-n:n) * dt)
        kernel = kernel/sum(kernel*dt)  # normalize (each spike contributes power 1)
        for j in -n:n  # go 4 sd each side
            if (k + j) >= 1 && (k + j) <= N    # in bounds 
                r[k+j] += kernel[n+j+1]
            end
        end
    end

    if padN>0
        t0 = t[padN]
    else
        t0 = 0.0
    end

    return (t[padN.+(1:sigN)].-t0, r[padN.+(1:sigN)])


end

# utility alias for GLR (because usually length(sd)==1)
function GLR(spiketime::Vector{Float64}, sd::Float64, dt::Float64, pad::Float64=0.0, T::Float64=maximum(spiketime))
    GLR(spiketime, [sd], dt, pad, T)
end
