# test Neuralian Toolbox

# test calls:

# test_Wald_sample(.013, 1.0)

# test_Exponential_sample(.001)

# test_Exwald_sample_sum(.013, 0.1, .01)

# test_Exwald_sample_sim(.013, 0.1, .01)

# inhomogenousPoisson_test(200)

# inhomogenousWald_test(1000)

# test_Exwald_Neuron_spontaneous(500, (.013, 0.1, .01))

# demo_Exwald_Neuron_sin(500, (.013, 0.1, .01), (.25, 1.0))

# test_Exwald_Neuron_phasedistribution(50000, (.013, 0.1, .01), (.25, 1.0), vec([45.0*i for i in 0:7]))    

# show_Exwald_Neuron_sineresponse((.013, 0.1, .01), (.25, 1.0), 64)
# show_Exwald_Neuron_sineresponse((.013, 0.1, .01), (.25, 1.0), 64, 0.05, 0.001)

# make Poisson spike train, listen and write to mp3
# I = zeros(100);Exponential_sample(I, .1);st = cumsum(I); listen(st); spiketimes2mp3(st, "aspiketrain")

#test_FractionalSteinhausenExwald_Neuron_phasedistribution(100000, .0, (.013, 1., .001), (40., 1.0), vec([45.0*i for i in 0:7]))

# blgParams = (.001, 4.0, 5.0)
# X = demo_blgStateMap(blgParams, .5, 128)
# heatmap!(X[1])

include("Neuralian_header.jl")


# compare interval distribution of inhomogenous Poisson process 
#   with rate r(t) = baserate + Δrate(t) to theoretical and fitted distributions.
# Intervals in an inhomogenous Poisson process should be Exponentially distributed
#  with mean interval equal to 1/(mean rate), i.e. same as a homogeneous Poisson process
#  with constant rate.
function check_inhomogeneous_Poisson_distribution()

    T = 60.    # simulate 1 minutes 
    dt = DEFAULT_SIMULATION_DT

    meanrate = 100.0                  # events per second
    Δrate = t -> meanrate*(t/T)^2     # quadratically increasing rate up to 2x initial

    # intervals by threshold crossing in time-varying Gaussian white noise
    ISI, R = time_varying_exponential_intervals_by_threshold_crossing(meanrate, Δrate, T, dt)

    # ISI histogram
    F = Figure()
    ax = Axis(F[1,1])
    ax.title = "Interval distribution of time-varying Poisson process is Exponential"

    # normalized histogram (probability density)
    # at least 32 bins or 500 intervals per bin
    H = fit(Histogram, ISI, nbins = max(Int(round(length(ISI)/32.0)), 32)) 

    # Frequency histogram edges and counts
    bin_edges = H.edges[1]
    bin_width = bin_edges[2]-bin_edges[1]
    bin_counts = H.weights./(sum(H.weights)/bin_width)
    bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)
    P_bin = bin_counts/sum(bin_counts)    # probability distribution 

    hist!(ax, ISI, bins=bin_edges, normalization = :pdf)

    # average rate
    t = 0.0:dt:T
    rhat = meanrate + sum(Δrate.(t))/length(t)

    # overlay expected distribution
    Exponential_expected = rhat.*exp.(-rhat*bin_centres)
    lines!(bin_centres, Exponential_expected, color = :salmon1, linewidth = 2, label = "Expected")

    # overlay fitted
    param, KLD_fitted = fit_Exponential(bin_centres, P_bin) # fit model
    Exponential_fitted = pdf(Exponential(param...), bin_centres)

    lines!(bin_centres, Exponential_fitted, linewidth = 2, color = :darkolivegreen, label = "Fitted")

    axislegend(ax)
    display(F)

    KLD_expected = KLD(bin_centres, bin_counts/sum(bin_counts), Exponential_expected/sum(Exponential_expected))

    return bin_counts, bin_centres, KLD_expected, KLD_fitted

end

# show that drift-diffusion first passage time simulation 
# generates intervals matching specified Wald distribution.
# Plots the ISI distribution overlaid with specified and fitted Wald models 
# The good news is that we get a pretty good fit to the specified model with only a few intervals.
function check_Wald_distribution(Waldparam::Tuple{Float64, Float64}, N::Int64 = 5000, 
    dt::Float64=DEFAULT_SIMULATION_DT)

    #  spontaneous spikes
    st = spiketimes_timevaryingWald(Waldparam, t->0.0, N, dt);


    F = Figure(size=(1600, 400))
    ax = Axis(F[1,1])

    ax.title = "Wald distribution from drift-diffusion model"

    ISI = diff(st)

    # frequency histogram 
    H = fit(Histogram, ISI, nbins = max(Int(round(N/500)), 32)) 

    # Frequency histogram edges and counts
    bin_edges = H.edges[1]
    bin_width = bin_edges[2]-bin_edges[1]
    bin_counts = H.weights./(sum(H.weights)*bin_width)
    bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)
    P_bin = bin_counts/sum(bin_counts)    # probability distribution 

    hist!(ax, ISI, bins=bin_edges, normalization = :pdf)

    # overlay pdf of specified distribution
    Wald_specified = pdf.(InverseGaussian(Waldparam...), bin_centres)
    lines!(bin_centres, Wald_specified, linewidth = 3, color = :salmon1, label = "Specified")

    # overlay fitted Wald
    param, KLD_fitted = fit_Wald(bin_centres, P_bin) # fit Wald model
    Wald_fitted = pdf(InverseGaussian(param...), bin_centres)
    lines!(bin_centres, Wald_fitted, linewidth = 1, color = :blue, label = "Fitted")
 
    axislegend(ax)
    display(F)

    #@infiltrate

    KLD_specified = KLD(bin_centres, bin_counts/sum(bin_counts), Wald_specified/sum(Wald_specified))

    return bin_counts, bin_centres, KLD_specified, KLD_fitted

end

# show that Exwald neuron model with parameters (μ, λ, τ) generates intervals 
# with distribution Exwald(μ, λ, τ) when input (cupula deflection) is 0.0.
# Plots the ISI distribution overlaid with specified and fitted  models 
function check_Exwald_neuron_spontaneous_distribution(EXWparam::Tuple{Float64, Float64,Float64}, N::Int64 = 5000, 
    dt::Float64=DEFAULT_SIMULATION_DT)


    exwald_neuron = make_Exwald_neuron(EXWparam, dt)

    # input = cupula deflection = 0.0
    δ = t-> 0.0

    # generate N intervals 
    ISI  = interspike_intervals(exwald_neuron, δ, N, dt)



    F = Figure(size=(1600, 400))
    ax = Axis(F[1,1])

    ax.title = "Exwald distribution from Neuron model"

    # frequency histogram
    T = 1.5*maximum(ISI)  # longest interval 
    bw = T/max(Int(round(N/500)), 128)   # at least 32 bins
    bin_edges = 0.0:bw:T
    H = fit(Histogram, ISI, bin_edges) 

    # Frequency histogram edges and counts
    bin_counts = H.weights./(sum(H.weights)*bw)
    bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)
    P_bin = bin_counts/sum(bin_counts)    # probability distribution for KLD

    hist!(ax, ISI, bins=bin_edges, normalization = :pdf)

    # overlay pdf of specified distribution
    gw = 0.01*bw
    grid = collect( 0.0:gw:T )
    Exwald_specified = Exwaldpdf(EXWparam..., grid)
    lines!(grid, Exwald_specified, linewidth = 5, color = :salmon1, label = "Specified")

    # overlay fitted Exwald
    param, KLD_fitted = fit_Exwald(bin_centres, bin_counts) # fit Exwald model
    Exwald_fitted = Exwaldpdf(param..., grid)
    lines!(grid, Exwald_fitted, linewidth = 5, color = :orchid4, label = "Fitted")
 
    axislegend(ax)
    display(F)

    KLD_specified = KLD(bin_centres, P_bin, Exwaldpdf(EXWparam..., bin_centres, true))

    return EXWparam, param, KLD_specified, KLD_fitted

end

# test Exwald sample generated by adding samples from Exponential and Wald distributions
function test_Exwald_sample_sum(mu, lambda, tau)

    N = 5000  # sample size
    I = Exwald_sample_sum(N, mu, lambda, tau)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau)            # threshold for Poisson component using same noise
    
    F = Figure()
    ax = Axis(F[1,1])
    ax.title = "Exwald Sample by sum of Exponential and Wald Samples"

    hist!(I, normalization=:pdf, bins=100)
    #    text!(ax, text = @sprintf "Simulation (%.5f, %.5f, %.5f, %.2f)" v s barrier trigger)

    dt = 1.0e-5
    T = maximum(I) * 1.2
    t = collect(dt:dt:T)
    lw = 2.5

    X = Exwaldpdf(mu, lambda, tau, t)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    lines!(t, W, linewidth = 2.0)

    lines!(t, P, linewidth = 2.0)
    #   text!(ax, text = @sprintf "Exponential (%.5f)" tau)
    lines!(t, X, linewidth = 2.0)
    #  text!( text = @sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)

    ymax = max(maximum(X), maximum(W)) * 1.25
    ylims!(0.0, ymax )
    xlims!(0.0, T)

    text!(ax, T/2.0, 0.8*ymax, text = @sprintf "μ = %5f " mu)
    text!(ax, T/2.0, 0.75*ymax, text = @sprintf "λ =  %5f" lambda)
    text!(ax, T/2.0, 0.7*ymax, text = @sprintf "τ =  %5f" tau)
    #@infiltrate

    display(F)
end


# # test Exwald sample generated by simulating threshold trigger & drift-diffusion FPT
function test_Exwald_sample_sim(mu, lambda, tau)

    N = 5000  # sample size
    I =  Exwald_sample_sim(N, mu, lambda, tau)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau)            # threshold for Poisson component using same noise
    
    F = Figure()
    ax = Axis(F[1,1])
    ax.title = "Exwald Sample by Threshold Trigger & Drift-Diffusion First Passage Time"

    hist!(I, normalization=:pdf, bins=100)
    #    text!(ax, text = @sprintf "Simulation (%.5f, %.5f, %.5f, %.2f)" v s barrier trigger)

    dt = 1.0e-5
    T = maximum(I) * 1.2
    t = collect(dt:dt:T)
    lw = 2.5

    X = Exwaldpdf(mu, lambda, tau, t)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    lines!(t, W, linewidth = 2.0)

    lines!(t, P, linewidth = 2.0)
    #   text!(ax, text = @sprintf "Exponential (%.5f)" tau)
    lines!(t, X, linewidth = 2.0)
    #  text!( text = @sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)

    ymax = max(maximum(X), maximum(W)) * 1.25
    ylims!(0.0, ymax )
    xlims!(0.0, T)

    text!(ax, T/2.0, 0.8*ymax, text = @sprintf "μ = %5f " mu)
    text!(ax, T/2.0, 0.75*ymax, text = @sprintf "λ =  %5f" lambda)
    text!(ax, T/2.0, 0.7*ymax, text = @sprintf "τ =  %5f" tau)
    #@infiltrate

    display(F)
end

# test dynamic Exwald with no stimulus
function test_Exwald_Neuron_spontaneous(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    (mu, lambda, tau) = Exwald_param
    spt = Exwald_Neuron_Nspikes(N, (mu, lambda, tau), t -> 0.0)


    F = Figure()
    ax = Axis(F[1,1])

    hist!(diff(spt), bins=64, normalization=:pdf)
    # label=@sprintf "Simulation")
    #xlims!(0.0, xlims(ax)[2])

    T = maximum(diff(spt))
    t = collect(0.0:dt:T)

    X = Exwaldpdf(mu, lambda, tau, t)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    lw = 2.5
    lines!(t, W, linewidth=lw)
    #    label=@sprintf "Wald (%.5f, %.5f)" mu lambda)
    lines!(t, P, linewidth=lw)
    # label=@sprintf "Exponential (%.5f)" tau)
    lines!(t, X, linewidth=lw * 1.5)
    # label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)

    ymax = max(maximum(X), maximum(W)) * 1.25
    ylims!(0.0, ymax)


    xlims!(0.0, T)
    # ylims!(0.0, 2.0 * maximum(X))
    ax.title = "Exwald Neuron Model Spontaneous ISI"

    text!(ax, T/2.0, 0.8*ymax, text = @sprintf "μ = %.5f" mu)
    text!(ax, T/2.0, 0.75*ymax, text = @sprintf "λ = %.5f" lambda)
    text!(ax, T/2.0, 0.7*ymax, text = @sprintf "τ = %.5f" tau)
    
    display(F)


end

# test dynamic Exwald with no stimulus
# SLIFparam = (v0, sigma, tau)
function test_SLIF_Neuron_spontaneous( N::Int64,
                                       SLIFparam::Tuple{Float64,Float64,Float64}, 
                                       pinkness::Int64=8, 
                                       dt::Float64=1.0e-3)

    SLIFneuron = make_SLIF_neuron(SLIFparam, pinkness, dt) 
    maxISI = 1000.0 # *dt
    ISI = interspike_intervals(SLIFneuron, t->0.0, N, maxISI, dt)

    #@infiltrate

    mu = mean(ISI)
    v = var(ISI)
    println("mu = ", mu, ", lambda = ", mu^3/v)
 



    F = Figure()
    ax = Axis(F[1,1])

    hist!(ISI, bins=64, normalization=:pdf)
    # label=@sprintf "Simulation")
    #xlims!(0.0, xlims(ax)[2])

    # T = maximum(diff(spt))
    # t = collect(0.0:dt:T)

    # X = Exwaldpdf(mu, lambda, tau, t)
    # W = pdf(InverseGaussian(mu, lambda), t)
    # P = exp.(-t ./ tau) ./ tau

    # lw = 2.5
    # lines!(t, W, linewidth=lw)
    # #    label=@sprintf "Wald (%.5f, %.5f)" mu lambda)
    # lines!(t, P, linewidth=lw)
    # # label=@sprintf "Exponential (%.5f)" tau)
    # lines!(t, X, linewidth=lw * 1.5)
    # # label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)

    # ymax = max(maximum(X), maximum(W)) * 1.25
    # ylims!(0.0, ymax)


    # xlims!(0.0, T)
    # ylims!(0.0, 2.0 * maximum(X))
    ax.title = "SLIF Neuron Model Spontaneous ISI"

    # text!(ax, T/2.0, 0.8*ymax, text = @sprintf "μ = %.5f" mu)
    # text!(ax, T/2.0, 0.75*ymax, text = @sprintf "λ = %.5f" lambda)
    # text!(ax, T/2.0, 0.7*ymax, text = @sprintf "τ = %.5f" tau)
    
    display(F)


end


# demo inhomogenous Exwald neuron with sinusoidal stimulus
# stimulus parameters Stim_param = (A, F), A = amplitude, F = frequency (Hz)
# NB stimulus amplitude is plotted x10 for visibility
function demo_Exwald_Neuron_sin(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64}, Stim_param::Tuple{Float64,Float64}, dt::Float64=DEFAULT_SIMULATION_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron
    spt = Exwald_Neuron_Nspikes(N, (mu, lambda, tau), stimulus)

   # Gaussian rate estimate
   gdt = 1.0e-3                      # sample interval for GLR estimate
   (t, r) = GLR(spt, [0.05], gdt)       # rate estimate r at sample times t

    #@infiltrate

    # compute expected mean rate (1/mean interval) during stimulus
    timevarying_τ = [PoissonTau_from_ThresholdTrigger(v.+stimulus(tt), s, trigger, DEFAULT_SIMULATION_DT) for tt in t]
    timevarying_μ = [Wald_parameters_from_FirstpassageTimeModel(v.+stimulus(tt), s, barrier)[1] for tt in t]
    timevarying_rate = 1.0./(timevarying_μ+timevarying_τ)

    Fig = Figure(size=(1200, 400))
    ax = Axis(Fig[1,1])

    # plot spike train
    splot!(ax, spt, 20.)

    # plot rate estimate 
    lines!(t, r, linewidth=2.5)

    # expected rate
    lines!(t,timevarying_rate, linewidth = 2.5 )

    # plot spontaneous level
    #lines!([t[1], t[end]], v * [1.0, 1.0])
    # plot stimulus
    lines!(t, [10.0*stimulus(x) for x in t], color = :salmon1, linewidth = 2.5)

    display(Fig)

    return spt
end


# demo dynamic Steinhausen-Exwald neuron with sinusoidal stimulus
# frequency f (Hz), maxw = peak angular velocity 
# >>>>>>>NB stimulus amplitude is maximum angular displacement in degrees<<<
#  because its easier to visualize the movement this way, e.g. wobbles +-10 deg at 1Hz
function deprecated_demo_Fractional_Steinhausen_Exwald_Neuron_sin(
        q::Float64, EXWparam::Tuple{Float64,Float64,Float64}, 
        f::Float64, maxw::Float64, T::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # stimulus amplitude
    # convert from max degrees displacement to radians/s^2 max angular acceleration
    #A = maxAngle * (π/180.0)*4.0*π^2*f^2  

    # # extract parameters of Exwald model
    # (mu, lambda, tau) = Exwald_param
    # (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    # trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # angular velocity at time t
    w = t -> maxw*sin(2π*f*t)
    # angular acceleration
    wdot = t -> diffcd(w, t, dt)

    # simulate spike train from fractional Steinhausen-Exwald neuron
    spt = fractionalSteinhausenExwald_Neuron(q, EXWparam, w, T)

   # Gaussian rate estimate
   # Matched filter for stimulus frequecy
   gdt = 1.0e-3                       # sample interval for GLR estimate
   s = 1.0/(2π*f);                    # kernel matched to stimulus period
   (tr, r) = GLR(spt, [s], gdt)       # rate estimate r at sample times t

    #@infiltrate

    # # # compute expected mean rate (1/mean interval) during stimulus
    # cupulaModel = create_cupula_state_update()
    # Gain = 10000.0  # spikes per second per radian deflection
    # timevarying_τ = [PoissonTau_from_ThresholdTrigger(v.+Gain*cupulaModel(stimulus(tt))[2], s, trigger, DEFAULT_SIMULATION_DT) for tt in tr]
    # timevarying_μ = [Wald_parameters_from_FirstpassageTimeModel(v.+Gain*cupulaModel(stimulus(tt))[2], s, barrier)[1] for tt in tr]
    # timevarying_rate = 1.0./(timevarying_μ+timevarying_τ)

    Fig = Figure(size=(1800, 400))
    ax = Axis(Fig[1,1])

    # plot spike train
    splot!(ax, spt, 20.)

    # plot rate estimate 
    lines!(tr, r, linewidth=2.5, color = :blue)

    # plot angular velocity and acceleration 
    w_tr =  w.(tr)
    wdot_tr = wdot.(tr)

    # scale factor to match stimulus amplitude to response amplitude on plot
    # (because we want to visually compare phase angles & don't care about amplitudes)
    Scale = sqrt(var(r)/var(wdot_tr))
    #print("Hello:"); println(Scale)

    # plot angular acceleration stimulus
    lines!(tr, sqrt(var(r)/var(wdot_tr))*wdot_tr, color = :salmon1, linewidth = 2.5)

     # plot angular velocity stimlus
    lines!(tr, sqrt(var(r)/var(w_tr))*w_tr, color = :green, linewidth = 2.5)    

    xlims!(0.0, tr[end])
    ax.title = "Fractional Steinhausen-Exwald Neuron "
    display(Fig)

    # return spike times, rate sample times, rate (at sample times), 
    # angular velocity and acceleration (at sample times)
    return (spt, collect(tr), r, w_tr, wdot_tr)
end

# demo dynamic Steinhausen-Exwald neuron with sinusoidal stimulus
# frequency f (Hz), maxw = peak angular velocity 
function demo_Fractional_Steinhausen_Exwald_Neuron_sin(
        q::Float64, EXWparam::Tuple{Float64,Float64,Float64}, 
        f::Float64, maxw::Float64, T::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # fractional torsion pendulum
    # default bandwidth of fractional approximation is .01-20.0 Hz
    # iniitalized in state (0.0,0.0)
    fpt = make_fractional_Steinhausen_stateUpdate_velocity_fcn(q, 0.0, 0.0)

    # Exwald neuron
    xneuron, _  = make_Exwald_neuron(EXWparam)

    # angular velocity
    w = t -> maxw*sin(2π*f*t)


    # NB1: fpt(w,t) returns updated cupula deflection δ given angular velocity w at time t, 
    #      from fractional torsion pendulum model.  It maintains internal state (δ, δ').
    # NB2: fpt(w,t) updates the state from t-dt to t, it doesn't give state at arbitrary t.

    # simulate spike train 
    #spt = fractionalSteinhausenExwald_Neuron(q, EXWparam, w, T)
    (spt, _) = spiketimes(xneuron, t->fpt(w, t), T)

   # Gaussian rate estimate
   # Matched filter for stimulus frequecy
   gdt = 1.0e-3                       # sample interval for GLR estimate
   s = 1.0/(2π*f);                    # kernel matched to stimulus period
   (tr, r) = GLR(spt, [s], gdt)       # rate estimate r at sample times t

    #@infiltrate

    # # # compute expected mean rate (1/mean interval) during stimulus
    # cupulaModel = create_cupula_state_update()
    # Gain = 10000.0  # spikes per second per radian deflection
    # timevarying_τ = [PoissonTau_from_ThresholdTrigger(v.+Gain*cupulaModel(stimulus(tt))[2], s, trigger, DEFAULT_SIMULATION_DT) for tt in tr]
    # timevarying_μ = [Wald_parameters_from_FirstpassageTimeModel(v.+Gain*cupulaModel(stimulus(tt))[2], s, barrier)[1] for tt in tr]
    # timevarying_rate = 1.0./(timevarying_μ+timevarying_τ)

    Fig = Figure(size=(1800, 400))
    ax = Axis(Fig[1,1])

    # plot spike train
    splot!(ax, spt, 20.)

    # plot rate estimate 
    lines!(tr, r, linewidth=2.5, color = :blue)

    # plot angular velocity and acceleration 
    w_tr =  w.(tr)
    wdot_tr = diffcd.(w, tr)

    # scale factor to match stimulus amplitude to response amplitude on plot
    # (because we want to visually compare phase angles & don't care about amplitudes)
    Scale = sqrt(var(r)/var(wdot_tr))
    #print("Hello:"); println(Scale)

    # plot angular acceleration stimulus
    lines!(tr, sqrt(var(r)/var(wdot_tr))*wdot_tr, color = :salmon1, linewidth = 2.5)

     # plot angular velocity stimlus
    lines!(tr, sqrt(var(r)/var(w_tr))*w_tr, color = :green, linewidth = 2.5)    

    xlims!(0.0, tr[end])
    ax.title = "Fractional Steinhausen-Exwald Neuron "
    display(Fig)

    # return spike times, rate sample times, rate (at sample times), 
    # angular velocity and acceleration (at sample times)
    return (spt, collect(tr), r, w_tr, wdot_tr, Fig)
end

# demo dynamic Steinhausen-Exwald neuron with sinusoidal stimulus
# frequency f (Hz), maxw = peak angular velocity 
function demo_Fractional_Steinhausen_Exwald_Neuron_blg(
        q::Float64, EXWparam::Tuple{Float64,Float64,Float64}, 
        BLGparam::Tuple{Float64, Float64, Float64}, T::Float64, dt::Float64=DEFAULT_SIMULATION_DT)

    # fractional torsion pendulum
    # default bandwidth of fractional approximation is .01-20.0 Hz
    # iniitalized in state (0.0,0.0)
    fpta = make_fractional_Steinhausen_stateUpdate_acceleration_fcn(q, 0.0, 0.0)

    # Exwald neuron
    xneuron, _  = make_Exwald_neuron(EXWparam)

    # blg angular velocity
    w = make_blg_generator(BLGparam, 16)

    # cupula deflection function(al) of angular velocity 
    # w() returns (ω, ω') but fpt() wants a function of ω' so we use t-> w(t)[1]
    δ = t-> fpta(t->w(t)[2], t)[1]


    # NB1: fpt(w,t) returns updated cupula deflection δ given angular velocity w at time t, 
    #      from fractional torsion pendulum model.  It maintains internal state (δ, δ').
    # NB2: fpt(w,t) updates the state from t-dt to t, it doesn't give state at arbitrary t.

    # simulate spike train 
    (spt, velocity)  = spiketimes(xneuron, δ, T)

   # Gaussian rate estimate
   # Matched filter for stimulus frequecy
   gdt = 1.0e-3                       # sample interval for GLR estimate
   s = 1.0/(2π*BLGparam[2]);          # kernel matched to upper band limit
   (tr, r) = GLR(spt, [s], gdt)       # rate estimate r at sample times t

    #@infiltrate

    # # # compute expected mean rate (1/mean interval) during stimulus
    # cupulaModel = create_cupula_state_update()
    # Gain = 10000.0  # spikes per second per radian deflection
    # timevarying_τ = [PoissonTau_from_ThresholdTrigger(v.+Gain*cupulaModel(stimulus(tt))[2], s, trigger, DEFAULT_SIMULATION_DT) for tt in tr]
    # timevarying_μ = [Wald_parameters_from_FirstpassageTimeModel(v.+Gain*cupulaModel(stimulus(tt))[2], s, barrier)[1] for tt in tr]
    # timevarying_rate = 1.0./(timevarying_μ+timevarying_τ)

    Fig = Figure(size=(1800, 400))
    ax = Axis(Fig[1,1])

    # plot spike train
    splot!(ax, spt, 20.)

    # plot rate estimate 
    lines!(tr, r, linewidth=2.5, color = :blue)

    #

    # plot angular velocity and acceleration 
    tw = 0.0:dt:T
    acceleration = diffcd(velocity, dt)
    # w_tr =  w.(tr)
    # wdot_tr = diffcd.(w, tr)

    # scale factor to match stimulus amplitude to response amplitude on plot
    # (because we want to visually compare phase angles & don't care about amplitudes)
    Scale = sqrt(var(r)/var(acceleration))
    #print("Hello:"); println(Scale)

    # # plot angular acceleration 
    lines!(tw, sqrt(var(r)/var(acceleration))*acceleration, color = :salmon1, linewidth = 2.5)

    # plot angular velocity 
    lines!(tw, sqrt(var(r)/var(velocity))*velocity, color = :green, linewidth = 2.5)    

    xlims!(0.0, T)
    ax.title = "Fractional Steinhausen-Exwald Neuron "
    display(Fig)

    # return spike times, rate sample times, rate (at sample times), 
    # angular velocity and acceleration (at sample times)
    return (spt, collect(tr), r, velocity, acceleration)
end


# NB this works but has bugs in cosmetics TBD
# Plot ISI distribution for Exwald neuron at specified phases of sinusoidal stimulus
# overlay the model-predicted Exwald for each phase
#   0 <= phase <= 360
# N = number of spikes 
# Exwald_param = (mu, lambda, tau)
# Stim_param = (amplitude, frequency /Schwarz)
# Example call:
#  test_Exwald_Neuron_phasedistribution(500000, xwp, (.25, 1.0), vec([45.0*i for i in 0:7]))
function test_Exwald_Neuron_phasedistribution(N::Int64,
    EXWparam::Tuple{Float64,Float64,Float64},
    Stim_param::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    # extract parameters
    (mu, lambda, tau) = EXWparam
    (A, F) = Stim_param
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal head angular velocity
    omega(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron for N intervals (=N spikes since interval[1] ends at spiketime[1])
    # during sinusoidal head movement
    afferent = make_Exwald_neuron(EXWparam, dt)
    ISI = interspike_intervals(afferent, omega, N, dt)
    spt = cumsum(ISI)    # spike times 

    # rate estimate
    dt4glr = 0.01
    (tx, glrate) = GLR(spt, [0.1], dt4glr);

    Npts = 128
    R0 = 1.0
    NP = length(phaseAngle)
    FigRadius = 500.0

    Fig = Figure(size = (2.0*FigRadius, 2.0*FigRadius))

    # ax0 allows text to be placed anywhere on the figure
    ax0 = Axis(Fig[1,NP+1])
    xlims!(ax0, 0., 1.)
    ylims!(ax0, 0., 1.)
    text!(ax0, .14, .95, fontsize = 24,
    text = "Exwald neuron ISI distribution during sinusoidal angular acceleration")
    text!(ax0, .5, .73, fontsize = 12, align = (:center, :center), text = "Peak acceleration")
    text!(ax0, .05, .92, fontsize = 16, text = "Spontaneous Model Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .125, h0, fontsize = 16, text = @sprintf "μ = %0.4f" mu)
    text!(ax0, .125, h0-dh, fontsize = 16, text = @sprintf "λ = %0.4f" lambda)    
    text!(ax0, .125, h0-2*dh, fontsize = 16, text = @sprintf "τ = %0.4f" tau)
    text!(ax0, .125, h0-3*dh, fontsize = 14, 
    text = @sprintf "Rate = %0.1f, CV = %0.2f, CV* = %0.2f" 1.0/(mu+tau) CV_fromExwaldModel(mu, lambda, tau) CVStar_fromExwaldModel(mu, lambda, tau))

    text!(ax0, .65, .92, fontsize = 16, text = "Stimulus Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .7, h0, fontsize = 16, text = @sprintf "Amplitude = %0.2f (%.4f x drift)" A A/v0)
    text!(ax0, .7, h0-dh, fontsize = 16, text = @sprintf "Frequency = %0.1fHz" F)    
    hidedecorations!(ax0)

    # ax1 shows stimulus, spike train and rate
    ax1 = Axis(Fig[1, NP+2], title = "Stimulus & spikes during one cycle")
    xlims!(ax1, 0., 1.0/F)
    ax1.xticks = vec([-1])
    ax1_width = 0.75*FigRadius
    ax1_height = ax1_width/(1.0+sqrt(5))
    period = 1.0/F
    t1 = collect(0.0:dt4glr:period)
    lines!(ax1, t1, [omega(t) for t in t1], color = :black, linewidth = 1.0)   # plot 1 cycle of stimulus
    display(Fig)  # to compute axis limits
    splot!(ax1, spt[minimum(findall(spt.>period)):maximum(findall(spt.<2.0*period))] .- period,
          ylims(ax1)[2]/4.0, 1.0, :navyblue)   

    # ax2 shows rate in every cycle except first and last (ignore edge effects in GLR)
    Nperiods = Int(min(floor(maximum(spt)/period), 128))
    ax2 = Axis(Fig[1, NP+3], title = @sprintf "Rate (%0.0f cycles)" Nperiods)
    xlims!(ax2, 0.0, 1.0/F)

    ns = Int(period/dt4glr)  # number of glr samples in period
    averageRate = zeros(ns+1)
    for i in 2:Nperiods
        averageRate = averageRate + glrate[((i-1)*ns).+(0:ns)]
        lines!(ax2, t1, glrate[((i-1)*ns).+(0:ns)], linewidth = 0.5)
    end
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 4.0, color = :white)
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 2.0, color = :black)

    insetPlotWide = 160.0
    insetPlotHigh = 160.0
    R = 0.7*FigRadius #  plot circle radius relative to width of parent axes
    spi = 1  # subplot index counter
    ax = []
    x0 = zeros(NP)
    y0 = zeros(NP)   
    xbig = -99.0   # for computing the x-axis limit, = 2x largest mean of fitted models in cycle
    maxT = -99.0
    maxy = -99.0
    fitted_param = zeros(3, NP)   # holds fitted Exwald parameters in columns
    D_kl = zeros(NP)    
    for i in 1:NP

        phaseRadians = -phaseAngle[i] * pi / 180.0

        # intervals at this phase angle
        ISI_i = intervalPhase(spt, phaseAngle[i], F, true)

        ax = (ax[:]..., Axis(Fig[1,i], xlabel = @sprintf "%.0f° " phaseAngle[i] ))
        x0[i] = round(FigRadius - R * cos(phaseRadians) - insetPlotHigh/2.0)       
        y0[i] = round(FigRadius - R * sin(phaseRadians) - insetPlotHigh/2.0)

        # frequency histogram
        T = 1.25*maximum(ISI_i)  # longest interval 
        if T>maxT
            maxT = T
        end
        bw = maxT/32; #max(Int(round(N/500)), 32)   # at least 32 bins
        bin_edges = 0.0:bw:maxT
        t = collect(0.0:(bw/10):maxT)
        H = fit(Histogram, ISI_i, bin_edges) 

        # Frequency histogram edges and counts
        bin_counts = H.weights./(sum(H.weights)*bw)
        bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)

        # fit Exwald Model
        fitted_param[:,i], D_kl[i] = fit_Exwald(bin_centres, bin_counts)

        thisbig = fitted_param[1,i] + fitted_param[3,i]   # mean = mu + tau
        if thisbig > xbig
            xbig = thisbig
        end

        # fit and display pdf
        fittedPDF = Exwaldpdf(fitted_param[:,i]..., t)
        if maximum(fittedPDF) > maxy
            maxy = maximum(fittedPDF)
        end
        lines!(ax[i], t,  fittedPDF, color = :red4, linewidth = 2.5)
        stairs!(bin_centres, bin_counts, color = :salmon)
        ax[i].backgroundcolor=:seashell
          
    end # phase angles

 

    display(Fig)  

    xmax = -99.0
    ymax = -99.0
    for i in 1:NP
        xmax = max(xmax, xlims(ax[i])[2])
        ymax = max(ymax, ylims(ax[i])[2])
    end

    xbig = 3.0*round(100.0*xbig)/100.0
    # if xbig<.1
    #     xbig = 2.0*round(100.0*xbig)/100.0
    # end
    # if xbig < .01  # don't expect this to happen. 
    #     xbig = .01
    # end
    for i in 1:NP
        xlims!(ax[i], 0.0, xbig) #xmax)
        ax[i].xticks = vec([-1])
        ylims!(ax[i], 0.0, ymax)
        ax[i].yticks = vec([-1])
    end

    # show fitted parameters
    h0 = 0.8
    dh = .1
    for i = 1:NP 
        text!(ax[i], xbig/2.0, h0*ymax, text = @sprintf "μ = %.4f" fitted_param[1,i])
        text!(ax[i], xbig/2.0, (h0-dh)*ymax, text = @sprintf "λ = %.4f" fitted_param[2,i])
        text!(ax[i], xbig/2.0, (h0-2.0*dh)*ymax, text = @sprintf "τ = %.4f" fitted_param[3,i])
    end

    #@infiltrate
    # plots are complete
    scalebarx = xlims(ax[1])[2]*[5/8, 7/8]  # scalebar endpoints
    scalebarlen = round(diff(scalebarx)[]*1000.)
    lines!(ax[1], scalebarx, 0.2*ymax*[1.0, 1.0], color = :black, linewidth = 2)

    text!(ax[1], mean(scalebarx), .25*ymax, text = (@sprintf "%0.0fms" scalebarlen), align = (:center,:center))

    # plots are complete, now move them into position
    for i in 1:NP
        setAxisBox(ax[i], x0[i], y0[i], insetPlotWide, insetPlotHigh)
    end
    setAxisBox(ax0, 0.0, 0.0, 2.0*FigRadius, 2.0*FigRadius)
    setAxisBox(ax1, FigRadius - ax1_width/2.0, FigRadius + 0.1*ax1_height,
                    ax1_width, ax1_height)
    setAxisBox(ax2, FigRadius - ax1_width/2.0, FigRadius - 1.1*ax1_height,
                    ax1_width, ax1_height)

    display(Fig)

    save("Exwald_Neuron_PhaseISI.png", Fig)

   return fitted_param, D_kl

end

# Plot ISI distribution for Exwald neuron at specified phases of sinusoidal stimulus
# overlay the model-predicted Exwald for each phase
#   0 <= phase <= 360
# N = number of spikes 
# EXW_param = (mu, lambda, tau)
# Stim_param = (amplitude, frequency /Schwarz)
# Example call:
#  demo_FractionalSteinhausenExwald_Neuron_phasedistribution(500, q, EXWparam, (.25, 1.0), vec([45.0*i for i in 0:7]))
function demo_FractionalSteinhausenExwald_Neuron_phasedistribution(N::Int64, q::Float64, 
    EXWparam::Tuple{Float64,Float64,Float64},
    Stimparam::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    # # extract parameters
    (mu, lambda, tau) = EXWparam
    (A, F) = Stimparam
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    # trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # fractional torsion pendulum with velocity input==true
    # default bandwidth of fractional approximation is .01-20.0 Hz
    # iniitalized in state (0.0,0.0)
    ftp = make_fractional_torsion_pendulum_stateUpdate(q, 0.0, 0.0, true, dt) 

    # Exwald neuron
    xneuron, _ = make_Exwald_neuron(EXWparam)

    # angular velocity function of time
    w = t -> A*sin(2π*F*t)

    # N Exwald intervals by simulation
    ISI = interspike_intervals(xneuron, t->ftp(w, t), N)

    # spike times 
    spt = cumsum(ISI)

    # rate estimate
    dt4glr = 0.01
    bandwidth = 0.1
    (tr, firingrate) = GLR(spt, [bandwidth], dt4glr);

    # Npts = 128
    # R0 = 1.0
    NP = length(phaseAngle)
    FigRadius = 500.0

    Fig = Figure(size = (2.0*FigRadius, 2.0*FigRadius))
 

    # ax0 allows text to be placed anywhere on the figure
    ax0 = Axis(Fig[1,NP+1], backgroundcolor = :snow2)
    xlims!(ax0, 0., 1.)
    ylims!(ax0, 0., 1.)
    text!(ax0, .5, .975, fontsize = FigRadius/21.0, align = (:center, :center),
     text = "Fractional Steinhausen-Exwald neuron ISI distributions during sinusoidal head movement")
    text!(ax0, .5, .73, fontsize = FigRadius/32.0, align = (:center, :center), text = "Peak Velocity")
    text!(ax0, .05, .92, fontsize = FigRadius/32.0, text = "Model Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .06, h0, fontsize = FigRadius/32.0, 
        text = @sprintf "(q, μ, λ, τ) = ( %0.2f, %0.3f,  %0.3f,  %0.3f)" q mu lambda tau)
    # text!(ax0, .125, h0-dh, fontsize = 16, text = @sprintf "λ = %0.4f" lambda)    
    # text!(ax0, .125, h0-2*dh, fontsize = 16, text = @sprintf "τ = %0.4f" tau)
    text!(ax0, .06, h0-dh, fontsize = FigRadius/32.0, 
    text = @sprintf "Rate = %0.1f, CV = %0.2f, CV* = %0.2f" 1.0/(mu+tau) CV_fromExwaldModel(mu, lambda, tau) CVStar_fromExwaldModel(mu, lambda, tau))

    text!(ax0, .65, .92, fontsize = FigRadius/32.0, text = "Stimulus Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .7, h0, fontsize = FigRadius/32.0, text = @sprintf "Amplitude = %0.2f (%.4f x drift)" A A/v0)
    text!(ax0, .7, h0-dh, fontsize = FigRadius/32.0, text = @sprintf "Frequency = %0.1fHz" F)    
    hidedecorations!(ax0)

    # ax1 shows stimulus, spike train and rate
    ax1 = Axis(Fig[1, NP+2], title = "Response to head angular velocity", backgroundcolor = :seashell)
    xlims!(ax1, 0., 1.0/F)
    ax1.xticks = vec([-1])
    ax1.titlesize = FigRadius/35.0
    ax1_width = 0.75*FigRadius
    ax1_height = ax1_width/(1.0+sqrt(5))
    period = 1.0/F
   # t1 = collect(0.0:dt4glr:period)
    lines!(ax1, tr, [w(t) for t in tr], color = :black, linewidth = 1.0)   # plot 1 cycle of stimulus
    display(Fig)  # to compute axis limits
    splot!(ax1, spt[minimum(findall(spt.>period)):maximum(findall(spt.<2.0*period))] .- period,
          ylims(ax1)[2]/4.0, 1.0, :navyblue)   

    # ax2 shows rate in every cycle except first and last (ignore edge effects in GLR)
    Nperiods = Int(min(floor(maximum(spt)/period), 128))
    ax2 = Axis(Fig[1, NP+3])
    ax2.title = @sprintf "Firing rate (%0.0f cycles)" Nperiods
    ax2.backgroundcolor = :seashell
    ax2.titlesize = FigRadius/40.0
    xlims!(ax2, 0.0, 1.0/F)

    ns = Int(period/dt4glr)  # number of glr samples in period
    averageRate = zeros(ns+1)
    t1 = (0:ns)*dt4glr
    for i in 2:Nperiods
        averageRate = averageRate + firingrate[((i-1)*ns).+(0:ns)]
        lines!(ax2, t1, firingrate[((i-1)*ns).+(0:ns)], linewidth = 0.5)
    end
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 4.0, color = :white)
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 2.0, color = :black)

    insetPlotWide = 0.36*FigRadius
    insetPlotHigh = 0.36*FigRadius

    R = 0.7*FigRadius #  plot circle radius relative to width of parent axes
    ax = []
    x0 = zeros(NP)
    y0 = zeros(NP)   
    fitted_param = Vector{Tuple{Float64, Float64, Float64}}(undef, NP)   # holds fitted Exwald parameters in columns
    D_kl = zeros(NP)              # holds KL divergence for fit at each phase
    maxy = -999.0
    maxT = -999.0
    for i in 1:NP

        phaseRadians = -phaseAngle[i] * pi / 180.0

        # intervals at specified phase
        ISI_i = intervalPhase(spt, phaseAngle[i], F, true)

        ax = (ax[:]..., Axis(Fig[1,i], xlabelpadding = -20, xlabel = @sprintf "%.0f° " phaseAngle[i] ))
        x0[i] = round(FigRadius - R * cos(phaseRadians) - insetPlotHigh/2.0)       
        y0[i] = round(FigRadius - R * sin(phaseRadians) - insetPlotHigh/2.0)

        # # predicted Distribution
        # v = v0 + w(2π*phaseAngle[i] / 360.0) # stimulus at phaseAngle[i]
        # (mu0, lambda0) = Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        # tau0 = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)
        # if (tau0==Inf) 
        #     tau0 = .1
        # end
   
        # frequency histogram
        T = 1.25*maximum(ISI_i)  # longest interval 
        if T>maxT
            maxT = T
        end
        bw = maxT/32; #max(Int(round(N/500)), 32)   # at least 32 bins
        bin_edges = 0.0:bw:maxT
        t = collect(0.0:(bw/10):maxT)
        H = fit(Histogram, ISI_i, bin_edges) 

        # relative frequency histogram 
        f_bin = H.weights./(sum(H.weights)*bw)
        bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)


        # fit Exwald Model
        fitted_param[i], D_kl[i] =  fit_Exwald(bin_centres, f_bin)

        # fit and display pdf
        fittedPDF = Exwaldpdf(fitted_param[i]..., t)
        if maximum(fittedPDF) > maxy
            maxy = maximum(fittedPDF)
        end
        lines!(ax[i], t,  fittedPDF, color = :red4, linewidth = 2.5)
        stairs!(bin_centres, f_bin, color = :salmon)
        ax[i].backgroundcolor=:seashell

    end # phase angles

    TT = Int(round(maxT*80.0))/100.0
    yLim = maxy*1.2
    for i in 1:NP
        xlims!(ax[i], 0.0, TT) #xmax)
        ax[i].xticks = vec([TT])
        ylims!(ax[i], 0.0, yLim)
        ax[i].yticks = vec([niceYtick(maxy, yLim)])
    end

    # show fitted parameters
    h0 = 1.05
    dh = .1
    hPos = maxT*0.45
    fontsize = FigRadius/50.0
    for i = 1:NP 
        text!(ax[i], hPos, h0*maxy, fontsize=fontsize, text = @sprintf "μ = %.4f" fitted_param[i][1])
        text!(ax[i], hPos, (h0-dh)*maxy, fontsize=fontsize, text = @sprintf "λ = %.4f" fitted_param[i][2])
        text!(ax[i], hPos, (h0-2.0*dh)*maxy, fontsize=fontsize, text = @sprintf "τ = %.4f" fitted_param[i][3])
    end

    # @infiltrate
    # plots are complete
    scalebarx = xlims(ax[1])[2]*[5/8, 7/8]  # scalebar endpoints
    scalebarlen = round(diff(scalebarx)[]*1000.)
    lines!(ax[1], scalebarx, 0.2*maxy*[1.0, 1.0], color = :black, linewidth = 2)

    text!(ax[1], mean(scalebarx), .25*maxy, text = (@sprintf "%0.0fms" scalebarlen), align = (:center,:center))

    # plots are complete, now move them into position
    for i in 1:NP
        setAxisBox(ax[i], x0[i], y0[i], insetPlotWide, insetPlotHigh)
    end
    setAxisBox(ax0, 0.0, 0.0, 2.0*FigRadius, 2.0*FigRadius)
    setAxisBox(ax1, FigRadius - ax1_width/2.0, FigRadius + 0.1*ax1_height, ax1_width, ax1_height)

    # seems to be a bug in Makie, can't reposition ax2 without redrawing
    display(Fig)
    setAxisBox(ax2, FigRadius - ax1_width/2.0, FigRadius - 1.1*ax1_height, ax1_width, ax1_height)

    display(Fig)
    save("FractionalSteinhausenExwald_Neuron_PhaseISI.png", Fig)

    return fitted_param, D_kl
end

# Interval distributions of Exwald neuron coupled to fractional torsion pendulum
# at specified phase angles during noisy sinusoidal head angular acceleration
function qTPExwald_phasedistribution_sinus(N::Int64, q::Float64, 
    EXWparam::Tuple{Float64,Float64,Float64},
    Stimparam::Tuple{Float64,Float64, Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    # # extract parameters
    (mu, lambda, tau) = EXWparam
    (A, F, S) = Stimparam   # amplitude, frequency and Gaussian noise s.d.
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    # trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # fractional torsion pendulum with velocity input==false (ie specify input = acceleration)
    # default bandwidth of fractional approximation is .01-20.0 Hz
    # iniitalized in state (0.0,0.0)
    ftp = make_fractional_torsion_pendulum_stateUpdate(q, 0.0, 0.0, false, dt) 

    # Exwald neuron
    xneuron = make_Exwald_neuron(EXWparam)

    # angular acceleration function of time, noisy sine wave
    w = t -> A*sin(2π*F*t) + S*randn()

    # N Exwald intervals by simulation
    ISI = interspike_intervals(xneuron, t->ftp(w, t), N)

    # spike times 
    spt = cumsum(ISI)

    # rate estimate
    dt4glr = 0.01
    bandwidth = 0.1
    (tr, firingrate) = GLR(spt, [bandwidth], dt4glr);

    # Npts = 128
    # R0 = 1.0
    NP = length(phaseAngle)
    FigRadius = 500.0

    Fig = Figure(size = (2.0*FigRadius, 2.0*FigRadius))
 

    # ax0 allows text to be placed anywhere on the figure
    ax0 = Axis(Fig[1,NP+1], backgroundcolor = :snow2)
    xlims!(ax0, 0., 1.)
    ylims!(ax0, 0., 1.)
    text!(ax0, .5, .975, fontsize = FigRadius/21.0, align = (:center, :center),
     text = "Fractional Steinhausen-Exwald neuron ISI distributions during sinusoidal head movement")
    text!(ax0, .5, .73, fontsize = FigRadius/32.0, align = (:center, :center), text = "Peak Velocity")
    text!(ax0, .05, .92, fontsize = FigRadius/32.0, text = "Model Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .06, h0, fontsize = FigRadius/32.0, 
        text = @sprintf "(q, μ, λ, τ) = ( %0.2f, %0.3f,  %0.3f,  %0.3f)" q mu lambda tau)
    # text!(ax0, .125, h0-dh, fontsize = 16, text = @sprintf "λ = %0.4f" lambda)    
    # text!(ax0, .125, h0-2*dh, fontsize = 16, text = @sprintf "τ = %0.4f" tau)
    text!(ax0, .06, h0-dh, fontsize = FigRadius/32.0, 
    text = @sprintf "Rate = %0.1f, CV = %0.2f, CV* = %0.2f" 1.0/(mu+tau) CV_fromExwaldModel(mu, lambda, tau) CVStar_fromExwaldModel(mu, lambda, tau))

    text!(ax0, .65, .92, fontsize = FigRadius/32.0, text = "Stimulus Parameters:")
    h0 = .9
    dh = .02
    text!(ax0, .7, h0, fontsize = FigRadius/32.0, text = @sprintf "Amplitude = %0.2f (%.4f x drift)" A A/v0)
    text!(ax0, .7, h0-dh, fontsize = FigRadius/32.0, text = @sprintf "Frequency = %0.1fHz" F)    
    hidedecorations!(ax0)

    # ax1 shows stimulus, spike train and rate
    ax1 = Axis(Fig[1, NP+2], title = "Response to head angular velocity", backgroundcolor = :seashell)
    xlims!(ax1, 0., 1.0/F)
    ax1.xticks = vec([-1])
    ax1.titlesize = FigRadius/35.0
    ax1_width = 0.75*FigRadius
    ax1_height = ax1_width/(1.0+sqrt(5))
    period = 1.0/F
   # t1 = collect(0.0:dt4glr:period)
    lines!(ax1, tr, [w(t) for t in tr], color = :black, linewidth = 1.0)   # plot 1 cycle of stimulus
    display(Fig)  # to compute axis limits
    splot!(ax1, spt[minimum(findall(spt.>period)):maximum(findall(spt.<2.0*period))] .- period,
          ylims(ax1)[2]/4.0, 1.0, :navyblue)   

    # ax2 shows rate in every cycle except first and last (ignore edge effects in GLR)
    Nperiods = Int(min(floor(maximum(spt)/period), 128))
    ax2 = Axis(Fig[1, NP+3])
    ax2.title = @sprintf "Firing rate (%0.0f cycles)" Nperiods
    ax2.backgroundcolor = :seashell
    ax2.titlesize = FigRadius/40.0
    xlims!(ax2, 0.0, 1.0/F)

    ns = Int(period/dt4glr)  # number of glr samples in period
    averageRate = zeros(ns+1)
    t1 = (0:ns)*dt4glr
    for i in 2:Nperiods
        averageRate = averageRate + firingrate[((i-1)*ns).+(0:ns)]
        lines!(ax2, t1, firingrate[((i-1)*ns).+(0:ns)], linewidth = 0.5)
    end
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 4.0, color = :white)
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 2.0, color = :black)

    insetPlotWide = 0.36*FigRadius
    insetPlotHigh = 0.36*FigRadius

    R = 0.7*FigRadius #  plot circle radius relative to width of parent axes
    ax = []
    x0 = zeros(NP)
    y0 = zeros(NP)   
    fitted_param = zeros(3, NP)   # holds fitted Exwald parameters in columns
    D_kl = zeros(NP)              # holds KL divergence for fit at each phase
    maxy = -999.0
    maxT = -999.0
    for i in 1:NP

        phaseRadians = -phaseAngle[i] * pi / 180.0

        # intervals at specified phase
        ISI_i = intervalPhase(spt, phaseAngle[i], F, true)

        ax = (ax[:]..., Axis(Fig[1,i], xlabelpadding = -20, xlabel = @sprintf "%.0f° " phaseAngle[i] ))
        x0[i] = round(FigRadius - R * cos(phaseRadians) - insetPlotHigh/2.0)       
        y0[i] = round(FigRadius - R * sin(phaseRadians) - insetPlotHigh/2.0)

        # # predicted Distribution
        # v = v0 + w(2π*phaseAngle[i] / 360.0) # stimulus at phaseAngle[i]
        # (mu0, lambda0) = Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        # tau0 = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)
        # if (tau0==Inf) 
        #     tau0 = .1
        # end
   
        # frequency histogram
        T = 1.25*maximum(ISI_i)  # longest interval 
        if T>maxT
            maxT = T
        end
        bw = maxT/32; #max(Int(round(N/500)), 32)   # at least 32 bins
        bin_edges = 0.0:bw:maxT
        t = collect(0.0:(bw/10):maxT)
        H = fit(Histogram, ISI_i, bin_edges) 

        # Frequency histogram edges and counts
        bin_counts = H.weights./(sum(H.weights)*bw)
        bin_centres = collect((bin_edges[2:end] + bin_edges[1:(end-1)])*0.5)


        # fit Exwald Model
        fitted_param[:,i], D_kl[i] = fit_Exwald(bin_centres, bin_counts)

        # fit and display pdf
        fittedPDF = Exwaldpdf(fitted_param[:,i]..., t)
        if maximum(fittedPDF) > maxy
            maxy = maximum(fittedPDF)
        end
        lines!(ax[i], t,  fittedPDF, color = :red4, linewidth = 2.5)
        stairs!(bin_centres, bin_counts, color = :salmon)
        ax[i].backgroundcolor=:seashell

    end # phase angles

    TT = Int(round(maxT*80.0))/100.0
    yLim = maxy*1.2
    for i in 1:NP
        xlims!(ax[i], 0.0, TT) #xmax)
        ax[i].xticks = vec([TT])
        ylims!(ax[i], 0.0, yLim)
        ax[i].yticks = vec([niceYtick(maxy, yLim)])
    end

    # show fitted parameters
    h0 = 1.05
    dh = .1
    hPos = maxT*0.45
    fontsize = FigRadius/50.0
    for i = 1:NP 
        text!(ax[i], hPos, h0*maxy, fontsize=fontsize, text = @sprintf "μ = %.4f" fitted_param[1,i])
        text!(ax[i], hPos, (h0-dh)*maxy, fontsize=fontsize, text = @sprintf "λ = %.4f" fitted_param[2,i])
        text!(ax[i], hPos, (h0-2.0*dh)*maxy, fontsize=fontsize, text = @sprintf "τ = %.4f" fitted_param[3,i])
    end

    # @infiltrate
    # plots are complete
    scalebarx = xlims(ax[1])[2]*[5/8, 7/8]  # scalebar endpoints
    scalebarlen = round(diff(scalebarx)[]*1000.)
    lines!(ax[1], scalebarx, 0.2*maxy*[1.0, 1.0], color = :black, linewidth = 2)

    text!(ax[1], mean(scalebarx), .25*maxy, text = (@sprintf "%0.0fms" scalebarlen), align = (:center,:center))

    # plots are complete, now move them into position
    for i in 1:NP
        setAxisBox(ax[i], x0[i], y0[i], insetPlotWide, insetPlotHigh)
    end
    setAxisBox(ax0, 0.0, 0.0, 2.0*FigRadius, 2.0*FigRadius)
    setAxisBox(ax1, FigRadius - ax1_width/2.0, FigRadius + 0.1*ax1_height, ax1_width, ax1_height)

    # seems to be a bug in Makie, can't reposition ax2 without redrawing
    display(Fig)
    setAxisBox(ax2, FigRadius - ax1_width/2.0, FigRadius - 1.1*ax1_height, ax1_width, ax1_height)

    display(Fig)
    save("FractionalSteinhausenExwald_Neuron_PhaseISI.png", Fig)

    return fitted_param, D_kl
end

# Track sinusoidal head movement given Exwald afferent neuron spike times
# using Bayesian secondary neurons in a circular map.
#  EXWphase is a matrix whose columns are Exwald parameters of afferent
#           interval distributions at specified phases during stimulus
#           generated by demo_FractionalSteinhausenExwald_Neuron_phasedistribution()
#           Phase angles are deduced from this by assuming equal spacing and 1st column = 0 degrees.
#  q is fractional order parameter (typically -0.1 - 0.5) used during learning interval distributions
#  Stimparam contains stimulus parameters (amplitude, frequency/Hz)  
#  Nspikes is total number of spikes to generate
function demo_map_sin(EXWparam::Tuple{Float64, Float64, Float64},
                      EXWphase::Matrix{Float64}, 
                      q::Float64, Stimparam::Tuple{Float64, Float64, Float64}, 
                      Nspikes::Int64=5000, dt::Float64=DEFAULT_SIMULATION_DT)
    
    M, N = size(EXWphase)
    @assert M = 3  "EXWphase must be 3xN matrix)"

    A, f, s = Stimparam        # stimulus amplitude, frequency and noise s.d. 
    phase = vec([2.0*pi*i/N for i in 0:(N-1)])   # N-vector of phase angles

    # time is "global" within this function
    t = 0.0

    # stimulus 
    omega(t_) = A*sin(2.0*pi*f*t_) + s*randn()

    # afferent neuron
    afferent = make_Exwald_neuron(EXWparam, dt)

    # Bayesian map (afferent 1 projects to all neurons in the map)
    afferent_projection = Vector{Vector{Int64}}(undef, N)
    for i in 1:N 
        afferent_projection[i] = [1]
    end
    map = make_Bayesian_neuron_map(EXWphase, afferent_projection)

    ringbuffer = ones(N)/N          # uniform distribution
    driftmean = 1.0/(f*N)           # mean drift time between buffer/neuron locations
    driftLambda = (2.0*pi/N)^2/s^2  # barrier height (distance) is 2pi/N

    # upate ring buffer by drift-diffusion given interval of length I
    function buffer_driftdiffuse(I)

        # this proportion of buffer[i] leaks to buffer[i+1]
        d = cdf(Inversegaussian(driftmean, driftLambda), I)

        temp = ringbuffer[N]
        for i = N:-1:2

            # transfer probability mass from location i to i-1
            ringbuffer[i-1] += d*ringbuffer[i]
            ringbuffer[i] *= d
        end
        ringbuffer[1] += d*temp
 
    end

    



    
end



function show_Exwald_Neuron_sineresponse( Exwald_param::Tuple{Float64,Float64,Float64},
    Stim_param::Tuple{Float64,Float64}, Nperiods::Int = 64,
    sd::Float64 = -1.0, 
    dt::Float64 = -1.0)


    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param

    # default GLR filter bandwidth 10% of period
    if sd < 0.0
        sd = 0.025/F
    end

    # default sampling interval smalller of .1 filter width or .1 tau
    if dt < 0.0
        dt = 0.1
        while dt > min(sd, tau)
            dt = 0.1*dt
        end
        dt = 1.0/Int(round(1.0/dt))   # fix numerical error in divide-by-10
    end


    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    Period = 1.0/F

    # simulate Exwald neuron for 131 cycles (1st 2 & last 1 will be discarded)
    spt = Exwald_Neuron(Float64(Nperiods+3)*Period, (mu, lambda, tau), stimulus)

    # rate estimate
    #dt4glr = 0.1
    (tx, glrate) = GLR(spt, Period/16.0, dt);



    R0 = 1.0

    Fig = Figure(size = (1200, 600))

    # ax0 allows text to be placed anywhere on the figure
    ax1 = Axis(Fig[1,1])
    ax2 = Axis(Fig[2,1])


    period = 1.0/F
    t1 = collect(0.0:dt:period)
    lines!(ax1, t1, [stimulus(t) for t in t1], color = :black, linewidth = 1.0)   # plot 1 cycle of stimulus
    display(Fig)
    # plot spike train over 2nd cycle
    splot!(ax1, spt[minimum(findall(spt.>2.0*period)):maximum(findall(spt.<3.0*period))] .- 2.0*period,
          ylims(ax1)[2]/4.0, 1.0, :navyblue)   


    ns = Int(round(period/dt))  # number of glr samples in period
    averageRate = zeros(ns+1)

    for i in 2 .+ (1:Nperiods)
        averageRate = averageRate + glrate[((i-1)*ns).+(0:ns)]
      #  if i <= 130 # display maximum of 128 rate samples 
            lines!(ax2, t1, glrate[((i-1)*ns).+(0:ns)], linewidth = 0.5)
       # end
    end

    # plot instantaneous rate from GLR averaged over Nperiods, COLOR = BLACK
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 4.0, color = :white)
    lines!(ax2, t1, averageRate./Nperiods, linewidth = 2.0, color = :black)

    # fit sinewave to rate estimate, pest = (offset, amplitude, phase)
    (minf, pest, ret) = Fit_Sinewave_to_Spiketrain(spt, F, sd, dt)

    # plot fitted sinewave, COLOR = RED
    lines!(ax2, t1, sinewave(pest,F,t1), color = :white, linewidth = 4.0)
    lines!(ax2, t1, sinewave(pest,F,t1), color = :red, linewidth = 2)
    lines!(ax2, t1, pest[1].+(pest[2]/A).*[stimulus(t) for t in t1], color = :green, linewidth = 2.0)   # plot 1 cycle of stimulus

    xlims!(ax1, [0.0, period])
    xlims!(ax2, [0.0, period])   
    display(Fig)

    println(@sprintf("Fitted rate sinewave: offset = %.3f, amplitude = %.3f, phase = %.1f°",
        pest[1], pest[2], pest[3]*180.0/pi))

   (minf, pest)

end


# test fit stationary Exwald distribution to spontaneous Exwald neuron model
# N = sample size
function test_fit_Exwald_neuron_stationary(N::Int64, mu::Float64, lambda::Float64, tau::Base.Float64)


    # spontaneous interspike intervals 
    ISI =  diff(vcat([0.0], Exwald_Neuron_Nspikes(N, (mu, lambda, tau), t->0.0)))

    #@infiltrate

    (p, ret) = Fit_Exwald_to_ISI(ISI, [mu, lambda, tau]) #,  [0.1, 0.1, 0.1])

    # dt = 1.0e-3
    # T = maximum(ISI)
    # t = collect(dt:dt:T)
    # hist(ISI, normalize=:pdf)

    counts, f_bin, bin_edges, bin_centres = ISI_distribution(ISI, nbins=128)

    F = Figure()
    ax = Axis(F[1,1], title = "Exwald model fitted to SLIF intervals")

    stairs!(ax, bin_centres, f_bin,color = :maroon, label = "")

    T = maximum(bin_edges)
    bw = bin_edges[2] - bin_edges[1]
    t = collect( 0.0:(bw/5.0):T )

    lines!(t, Exwaldpdf(p[1], p[2], p[3], t), linewidth=2)
    lines!(t, Exwaldpdf(mu, lambda, tau, t), linewidth=2)

    display(F)
    p

end

# build a 2D histogram (map) of states (ω, ̇ω) visited by BLG stimulus
# returns the map as a matrix, visualize it using heatmap!() 
function demo_blgStateMap(blgParams::Tuple{Float64, Float64, Float64}, Δω::Float64 = 1.0, Nrep::Int64=4)

    stateMap = init_stateMap(blgParams, Δω)

    dt = .001
    T  = 1.0/blgParams[1]    # blg period



    for rep = 1:Nrep
        # new blg generator in every period
        blg = make_blg_generator(blgParams, 32)
        # simulate band-limited Gaussian white noise 
        t = 0.0
        while t<T
            state = blg(t)
            map_add(stateMap, state, 1.0, false)
            t += dt
        end
    end

    stateMap
end

# pdf of states and conditional spike intensity 
# fractional (q) Steinhausen Exwald (FSX) neuron with parameters EXWparam responding to blg stimulus 
# blgParams = (flower, fupper, rms) 
# nb BLG stimulus is periodic with period 1/flower, replicate over samples of this length 
# e.g.   
#  states, spikes = demo_FSX_blg_statemap((.01, 2.0, 25.0), 0.5, (0.01, 1.0, 0.01)), 16)
#  F = Figure(); ax1 = Axis(F[1,1]); heatmap!(states[1]); ax2 = Axis(F[1,2]); heatmap!(spikes[1]); 
function demo_FSX_blg_statemap(blgParams, q::Float64, EXWparam, Nrep)

    Δω = 2.0
    stateMap = init_stateMap(blgParams, Δω)  # frequency histogram/pdf of states
    spikeMap = deepcopy(stateMap)            # frequency histogram/pdf of spikes

    dt = DEFAULT_SIMULATION_DT
    T  = 1.0/blgParams[1]    # blg period

    # fractional torsion pendulum model ftp(w,t)
    #   updates cupula deflection δ from t-dt to t given head angular acceleration w at t 
    # Has internal state (δ, δ') iniitalized to (0.0, 0.0)
    #     and M=2N+1 auxiliary state variables for fractional derivative approximation
    ftp = make_fractional_Steinhausen_stateUpdate_acceleration_fcn(q, 0.0, 0.0)

    # Exwald neuron
    (xwneuron, _ ) = make_Exwald_neuron(EXWparam)



    for rep = 1:Nrep

        # new blg generator in every period
        w = make_blg_generator(blgParams, 4)

        # dynamic update cupula 
        δ = t-> ftp(t->w(t)[2], t)[1]

        # simulate 
        t = 0.0
        while t<T

            # increment state frequency histogram (state map) at current stimulus state (ω, ̇ω) 
            map_add(stateMap, w(t), 1.0, false)

            # update exwald neuron given cupula deflection (1st state variable)
            # returns true if the neuron fired
            if xwneuron(δ, t)==true  # neuron fired 
                map_add(spikeMap, w(t), 1.0, false) # record a spike occured in this state
               # println(t, ", ", w(t))
            end
            t += dt
        end
        println(rep)
    end
    (stateMap, spikeMap)
end

# demo Bayesian inference from spikes in simplest possible case 
# where the world is in one of two possible states (static)
# and there is one afferent neuron whose interval distribution differs between states.
# Here the world is in state 1.
function demo_Bayesian_infer_binary_state_from_spiketrain(T::Float64 = 1.0, dt::Float64=DEFAULT_SIMULATION_DT)

    EXWparam_1 =  (.01, 1.0, 0.001)      # afferent interval distribution in state 1
    EXWparam_2 =  (.01, 1.0, 0.00125)    # .. state 2

    # 1 sensory afferent neuron 
    afferent, _ = make_Exwald_neuron(EXWparam_1, dt)

    # 2 secondary Bayesian neurons 
    # Each secondary neuron has one compartment
    # neuron 1 likelihood corresponds to the afferent distribution
    # neuron 2 likelihood is slightly different.
    # We specify a vector of likelihood functions for each neuron, 
    #   in this 1-compartment case each vector has length 1 and contains a tuple defining the likelihood.
    # We specify connectivity from the afferent nerve as a vector of vectors containing indices of
    #   neurons that project to each secondary neuron. In this example there is one afferent neuron
    #   that projects to each secondary neuron.
    update_map = make_Bayesian_neuron_map([[EXWparam_1], [EXWparam_2]], [[1], [1]])

    # Simulate spontaneous spiking up to time T
    spt = spiketimes(afferent, t->0.0, T, dt)[1]
    N = length(spt)

    # Infer posterior distribution over world states at each spike time
    p = Array{Float64}(undef, N, 2)   # 2 states, N intervals
    for i in 1:N

        # update map
        p[i, :] = update_map(spt[i], 1)

    end

    F = Figure(size = (1200, 600))

    ax1 = Axis(F[1,1], title = "Spike distributions")
    D = 1.5*maximum(diff(spt))   # 1.5 x largest interval
    t = collect((1:1000)*D/1000.0)
    lines!(ax1, t, Exwaldpdf(EXWparam_1..., t))
    lines!(ax1, t, Exwaldpdf(EXWparam_2..., t))

    ax2 = Axis(F[1,2], title = "Inference from spikes", ylabel = "posterior probability")
    lines!(ax2, spt, p[:,1])
    lines!(ax2, spt, p[:,2])    
    splot!(ax2, spt, .1)

    display(F)

    #save("BinaryInference.png", F)

    return F

end


# Fit Exwald model to ISI distribution of SLIF neuron spontaneous firing 
# SLIF model is dV   = (a + u(t) - V[] / tau) * dt + σ_v * sqrt_dt * randn()
# in the limit of large tau this is a drift-diffusion process with Inverse Gaussian ISI distn 
# SLIFparam = (mu, lambda, tau) where (mu, lambda) are parameters of the limiting IG distn  
function demo_fit_Exwald2SLIF(Param::Tuple{Float64, Float64, Float64}, N::Int64=1000,
      dt::Float64=DEFAULT_SIMULATION_DT)

    # get SLIF model update function and SLIF model coeffs
    SLIFneuron, SLIFparam = make_SLIF_neuron(Param...) 

    # simulate ISIs
    ISI = interspike_intervals(SLIFneuron, t->0.0, N) 

    #ISI = quantize_intervals(ISI)  # 300us quantization (sample resolution for real data)

    # fit Exwald distribution to ISI Distribution
    fitted_EXWparam, goodness_of_fit = Fit_Exwald_to_ISI(ISI) 

    counts, f_bin, bin_edges, bin_centres = ISI_distribution(ISI,nbins = 64)
    bw = bin_edges[2] - bin_edges[1]

    F = Figure()
    ax = Axis(F[1,1], title = "Exwald model fitted to SLIF intervals")

    SLIF_label = @sprintf "SLIF: a = %.5f, σ_v = %.5f, τ = %.5f" SLIFparam[1] SLIFparam[2] SLIFparam[3]
    stairs!(bin_centres.+bw/2, f_bin,color = :maroon, label = SLIF_label)
    grid = collect( 0.0:(bw/10.0):maximum(bin_edges) )
    EXW_label = @sprintf "Exwald: μ = %.5f, λ = %.5f, τ = %.5f" fitted_EXWparam[1] fitted_EXWparam[2] fitted_EXWparam[3]
    lines!(grid, Exwaldpdf(fitted_EXWparam..., grid), linewidth=2, color = :blue, label = EXW_label)

    axislegend(ax, position = :rt)

    display(F)

    # return fitted parameters and figure handle
    return fitted_EXWparam, F

end


# Overlay plot of spiking neuron model and theoretical spontaneous ISI distribution
# usage: see compare_Wald_neuron_ISI_to_Wald_pdf()
function compare_ISI_to_pdf(neuron::Function, pdf::Function, 
        N::Int64=1000, timeout::Float64 = 1.0, 
        dt::Float64=DEFAULT_SIMULATION_DT)

    ISI = interspike_intervals(neuron, _->0.0, N, timeout)

    # ISI histogram
    counts, f_bin, bin_edges, bin_centres = ISI_distribution(ISI,nbins = 64)
    bw = bin_edges[2] - bin_edges[1]

    # ISI stats
    meanISI = mean(ISI)
    sdISI = std(ISI)
    cvISI = sdISI/mean(ISI)
    cvStarISI = CVStar(ISI)

    # pdf model
    grid = collect( 0.0:(bw/10.0):(maximum(bin_edges)*1.25 ) )
    pdf_vec = [ pdf(t) for t in grid]
    meanPdf, sdPdf, cvPdf, cvstarPdf = pdf_stats(grid, pdf_vec)

    F = Figure(size = (800, 800))
    ax = Axis(F[1,1])
    maxf = maximum(f_bin)
    ylims!(0.0, maxf*1.25)

    # draw ISI histogram and pdf with dummy labels 
    # Labels will be completed by the caller, who knows the model parameters
    stairs!(bin_centres.+bw/2, f_bin,color = :maroon, label = "ISI")
    lines!(grid, pdf_vec, linewidth=2, color = :blue, label = "PDF")
 
    ISI_stats_label = @sprintf "ISI,  Mean:  %.5f, SD: %.5f, CV: %.5f, CV*: %.5f" meanISI sdISI cvISI cvStarISI    
    lines!(meanISI*[1.0, 1.0], [0.0, maxf/8.0], 
            linewidth = 2.0, linestyle = :dot, color = :maroon, label = ISI_stats_label)

    PDF_stats_label = @sprintf "PDF, Mean:  %.5f, SD: %.5f, CV: %.5f, CV*: %.5f" meanPdf sdPdf cvPdf cvstarPdf  
    lines!(meanPdf*[1.0, 1.0], [0.0, maxf/8.0], 
            linewidth = 2.0, linestyle = :dot, color = :blue, label = PDF_stats_label)
    
    axislegend(ax, position = :rt)

    display(F)

    # return figure handle
    return  F

end

# overlay plot of wald neuron ISI distribution and Wald pdf (theoretical ISI distribution)
# use this as a template for other neurons 
function compare_Wald_neuron_ISI_to_Wald_pdf(mu::Float64, lambda::Float64, N::Int64=1000)

    w_neuron, _ = make_Wald_neuron(mu, lambda, dt)
    wald_pdf = t->pdf(InverseGaussian(mu, lambda), t)

    # last parameter is timeout (usually simpler than this but Wald has wide range of shapes)
    F = compare_ISI_to_pdf(w_neuron, wald_pdf, N, mu+sqrt(1.0/lambda))

    ax = F.content[1]
    ax.title = "Wald neuron spontaneous ISI distribution and Wald PDF" 

    # In this example the neuron and the pdf have the same parameters 
    # but that is not true in general (e.g. qSLIF neuron vs. Exwald pdf)
    ax.scene.plots[1].label[] = @sprintf "Neuron mu = %.4f, lambda = %.4f" mu lambda 
    ax.scene.plots[2].label[] = @sprintf "PDF      mu = %.4f, lambda = %.4f" mu lambda 

    # update legend and redraw
    delete!(F.content[end])   # last entry is label (fingers crossed)
    axislegend(ax, position=:rt)
    display(F)

    return F

end


# overlay plot of Poisson neuron ISI distribution and Exponential pdf (theoretical ISI distribution)
function compare_Poisson_neuron_ISI_to_Exponential_pdf(tau::Float64, N::Int64=1000)

    p_neuron, _ = make_Poisson_neuron(tau)

    Exponential_pdf = t->pdf(Exponential(tau), t)

    # last parameter is timeout (p<1e-6)
    F = compare_ISI_to_pdf(p_neuron, Exponential_pdf, N, 14.0*tau)

    ax = F.content[1]
    ax.title = "Poisson neuron spontaneous ISI distribution and Exponential PDF" 

    # In this example the neuron and the pdf have the same parameters 
    # but that is not true in general (e.g. qSLIF neuron vs. Exwald pdf)
    ax.scene.plots[1].label[] = @sprintf "Neuron tau = %.4f" tau
    ax.scene.plots[2].label[] = @sprintf "PDF      tau = %.4f" tau

    # update legend and redraw
    delete!(F.content[end])   # last entry is label (fingers crossed)
    axislegend(ax, position=:rt)
    display(F)

    return F

end

# Fit Exwald model to ISI distribution of fractional SLIF neuron spontaneous firing 
# SLIF model is dV   = (a + u(t) - V[] / tau) * dt + σ_v * sqrt_dt * randn()
# in the limit of large tau this is a drift-diffusion process with Inverse Gaussian ISI distn 
# SLIFparam = (mu, lambda, tau) where (mu, lambda) are parameters of the limiting IG distn  
function demo_fit_Exwald2qSLIF(Param::Tuple{Float64, Float64, Float64}, q::Float64=1.0, 
        N::Int64=1000,
        dt::Float64=DEFAULT_SIMULATION_DT)

    fitted_EXWparam, goodness_of_fit, ISI, SLIFparam  = fit_Exwald_to_qSLIF(Param, q, N) 

    meanISI = mean(ISI)
    sdISI = std(ISI)
    cvISI = sdISI/mean(ISI)
    cvStar = CVStar(ISI)

    counts, f_bin, bin_edges, bin_centres = ISI_distribution(ISI,nbins = 64)
    bw = bin_edges[2] - bin_edges[1]

    F = Figure(size = (800, 800))
    ax = Axis(F[1,1], title = "Exwald model fitted to qSLIF neuron spontaneous interval distribution")
    maxf = maximum(f_bin)
    ylims!(0.0, maxf*1.25)

    SLIF_label = @sprintf "qSLIF: a = %.5f, σ_v = %.5f, τ = %.5f, q = %.2f" SLIFparam[1] SLIFparam[2] SLIFparam[3] q
    stairs!(bin_centres.+bw/2, f_bin,color = :maroon, label = SLIF_label)
    grid = collect( 0.0:(bw/10.0):maximum(bin_edges) )
    EXW_label = @sprintf "Exwald: μ = %.5f, λ = %.5f, τ = %.5f" fitted_EXWparam[1] fitted_EXWparam[2] fitted_EXWparam[3]
    lines!(grid, Exwaldpdf(fitted_EXWparam..., grid), linewidth=2, color = :blue, label = EXW_label)
    STATS_label = @sprintf "Mean:  %.5f, SD: %.5f, CV: %.5f, CV*: %.5f" meanISI sdISI cvISI cvStar
    lines!(meanISI*[1.0, 1.0], [0.0, maxf/8.0], 
            linewidth = 2.0, linestyle = :dot, color = :seagreen, label = STATS_label)

    axislegend(ax, position = :rt)

    display(F)

    #@infiltrate

    # return fitted parameters and figure handle
    return fitted_EXWparam, goodness_of_fit, F

end


# DEMO fit qSLIF neuron to Exwald model spontaneous ISI distribution 
function demo_fit_qSLIF2Exwald(mu::Float64, lambda::Float64, tau::Float64, 
        N::Int64=5000,
        dt::Float64=DEFAULT_SIMULATION_DT)

    # Fit qSLIF parameters to Exwald model
    qSLIFparam, objective = fit_qSLIF_to_Exwald_v2(mu, lambda, tau, N= N)

    @infiltrate

    # unpack
    (a, sigma, tau_memb, q) = qSLIFparam

    # construct this qSLIF neuron, unpack qSLIF coeffs
    neuron, _ = make_qSLIFc_neuron(qSLIFparam...)

    # generate spontaneous ISIs from qSLIF model
    ISI = interspike_intervals(neuron, t->0.0, N)

    meanISI = mean(ISI)
    sdISI = std(ISI)
    cvISI = sdISI/mean(ISI)
    cvStar = CVStar(ISI)

    counts, f_bin, bin_edges, bin_centres = ISI_distribution(ISI,nbins = 64)
    bw = bin_edges[2] - bin_edges[1]

    F = Figure(size = (800, 800))
    ax = Axis(F[1,1], title = "qSLIF neuron spontaneous interval distribution fitted to Exwald model")
    maxf = maximum(f_bin)
    ylims!(0.0, maxf*1.25)

    SLIF_label = @sprintf "qSLIF: a = %.5f, σ_v = %.5f, τ = %.5f, q = %.2f" a sigma tau_memb q
    stairs!(bin_centres.+bw/2, f_bin,color = :maroon, label = SLIF_label)
    grid = collect( 0.0:(bw/10.0):maximum(bin_edges) )
    EXW_label = @sprintf "Exwald: μ = %.5f, λ = %.5f, τ = %.5f" mu lambda tau
    lines!(grid, Exwaldpdf(mu, lambda, tau, grid), linewidth=2, color = :blue, label = EXW_label)

    lines!((mu+tau)*[1.0, 1.0], [0.0, maxf/8.0], 
            linewidth = 2.0, linestyle = :dot, color = :blue)
    ISI_label = @sprintf "ISI Mean:  %.5f, SD: %.5f, CV: %.5f, CV*: %.5f" meanISI sdISI cvISI cvStar
    lines!(meanISI*[1.0, 1.0], [0.0, maxf/8.0], 
            linewidth = 2.0, linestyle = :dot, color = :maroon, label = ISI_label)
    axislegend(ax, position = :rt)

    display(F)

    #@infiltrate

    # return fitted parameters and figure handle
    return qSLIFparam, objective, F

end

# plot cv vs efficiency (bits/watt) and channel capacity (bits/second)
# for Exwald neurons
function show_efficiency_vs_channelcapacity()

    N = 32
    cv = logrange(.05, 0.8, length = N)
    bits_spike = zeros(N)
    bits_second = zeros(N)
    ATP_second = zeros(N)

    for i in 1:N

        bits_spike[i], bits_second[i], ATP_second[i] = Exwald_entropy(Exwald_fromCV(cv[i]))

    end

    F = Figure()
    ax = Axis(F[1,1], xscale = log10,  xtickformat = "{:.4f}",
        xlabel = "CV", ylabel = "Information", title = "Information-Energy Tradeoff in Vestibular Afferent Spiking")
    ax.xticks = [.05, .1, .2, .5]

    lines!(cv, bits_spike, color = :red, label = "bits per spike")
    lines!(cv, .01*bits_second, color = :green, label = "bits per 10ms") # bits per 10ms
    lines!(cv, .25*ATP_second, color = :blue, label = "relative power consumption")

    axislegend(ax, position = :lb)

    display(F)

    return F  # bits_spike, bits_second, ATP_second

end

function map_channelCapacity_PowerConsumption_Exwald(mu::Float64)

    # parameter ranges from PP&H 2024
    NL = 64
    lambda = logrange(0.01, 100.0, length = NL)
    NT = 64
    tau    = logrange(.00001, .1, length = NT)

    # entropy of a spike (interval)
    S = NaN*ones(NT, NL)

    # channel capacity at rest 
    C =  NaN*ones(NT, NL)

    # power consumption at rest
    P = NaN*ones(NT, NL)

    for i in 1:length(tau)
        for j in 1:length(lambda)

            # spike entropy, channel capacity, power consumption
            S[i,j],C[i,j],P[i,j] = Exwald_entropy((mu, lambda[j], tau[i]))  

        end
    end

    F = Figure(size = (1200, 400))
    axS = Axis(F[1,1], xscale = log10, yscale = log10,                
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    axC = Axis(F[1,2], xscale = log10, yscale = log10,                
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")
    axP = Axis(F[1,3], xscale = log10, yscale = log10,                
                xtickformat = "{:.4f}", ytickformat = "{:.5f}")   
                
    heatmap!(axS, tau, lambda, S)
    heatmap!(axC, tau, lambda, C)
    heatmap!(axP, tau, lambda, P)

    display(F)

    return S, C, P, tau, lambda, F

end

# test fractional derivative operator
function test_dq(q::Float64=0.0, f::Float64=1.0, dt::Float64 = DEFAULT_SIMULATION_DT)

    # time
    T = 100.0
    t = collect(0.0:dt:T)

    # fractional derivative operator d^q
    # specified q, operator bandwidth in Hz, time step length
    dq = make_fractional_derivative(q)

    # input 
    u = sin.(2π*f.*t)

    # output
    y = zeros(length(t))
    
    for i in 1:length(t)

        y[i] = dq(u[i])

    end

    # pull out last  period (assuming steady state by then)
    period = 1.0/f
    iTail = Int64(round((T - period)/dt)):length(t)
    u = u[iTail]
    y = y[iTail]
    t = collect(1:length(iTail))*dt

    # Time of first peak in u and y 
    maxu = maximum(u)
    tpu = findfirst(u.==maxu)*dt
    maxy = maximum(y)
    tpy = findfirst(y.==maxy)*dt

    # Gain and phase
    Gain = maxy/maxu
    Phase = 360.0*(tpy-tpu)/period

    F = Figure(size=(1200, 600))
    ax = Axis(F[1, 1])

    # expected gain is  w^q, expected phase -90*q
    Ghat = (2π*f)^q 
    Phat = -90*q

    ax.title = @sprintf "Frequency: %.2f, Gain: %.3f (%.3f), Phase: %.3f (%.3f)" f Gain Ghat Phase Phat

    lines!(ax, t, u)
    lines!(ax, t, y)

    display(F)

end

# Bode plots of chinchilla torsion pendulum model
function demo_torsionpendulum()

    torsionpendulum = make_torsion_pendulum()

    BodePlot(torsionpendulum, 
        "Torsion Pendulum model of Chinchilla cupula displacement re angular velocity", 
        1.0, 1.0/60.0, 100., 32)

end

# plot of example qSLIF/dynamic Exwald neuron responding to sinusoidal input current 
# stimulus frequency f for default 4 cycles. Force f to be an integer, makes life easier.
# i selects neuron from 1 (very regular) to 5 (very irregular)
# preceded with and followed by N_spont (default f) periods of spontaneous activity
function plot_qSLIF_example(i::Int64, f::Int64=1, A::Float64 = 8.0, 
        Ncycles::Int64 = 4, Nrep::Int64 = 128, N_spont::Int64 = -1; 
        GLR_ON::Bool=true, SPIKEPLOT_ON::Bool=true)

    # cast f to Float64
    f = Float64(f)

    period = 1.0/f

    # stimulus duration 
    T_stim = Ncycles*period

    # delay before stimulus onset and wait time after stimulus offset 
    # default 1 second 
    if N_spont<0
        T_spont = 1.0
    else
        T_spont = N_spont*period 
    end

    # neuron descriptors
    neuronLabel = ( "Regular afferent (1)" ,
                    "Regular afferent (2)" ,
                    "Intermediate afferent (3)",
                    "Irregular afferent (4)",
                    "Irregular afferent (5)" 
                    )

    # add edges, to be cut off after filtering (to get rid of filter edge effect)
    T_edge = period 

    # time vector, 
    # allow for spontaneous + edges before and after stim
    T = T_stim +  2.0*(T_edge+T_spont)
    #println("T=", T)

    # GLR sample interval, 100 samples per stimulus cycle
    glr_dt = 0.01*period

    # glr sample times
    # precomputed outside loop because its the same every time
    t = collect(0.0:glr_dt:T)   

    # Stimulus is sine wave A*sin(2πft) 
    # with zero before L and after R
    stimulus(s) = (s>=(T_edge+T_spont)) && (s<=T-T_edge-T_spont) ? A*sin(2π*f*s) : 0.0

    # sample time indices with edges trimmed, for trimming GLR estimate
    # and trimmed sample times, for plotting edge-trimmed response
    t_index_trimmed = findall(s-> (s>=T_edge) && (s<=T-T_edge), t) 
    t_trimmed = t[t_index_trimmed] .- T_edge
    N_trimmed = length(t_trimmed)

    # construct the specified example neuron 
    neuron, param, summarystats = qSLIF_example_neuron(i)

    F = Figure(size = (1400, 400), fontsize=14)
    ax1 = Axis(F[1,1], title = neuronLabel[i])
    ylims!(0.0, 100.0)
    ax1.yticks = collect(0.0:10.0:100)
    ax1.ylabel = "Spikes per second"

    # will compute GLR every time but only show it 
    # if there are enough spikes per stimulus cycle on every rep for this to not be messy
    spikecount = 0.0  # count spikes after 1st cycle and to end of stimulus in each rep
    show_GLR = true
    minSpikeCount = 2.0

    # start and end time for extracting spikes for Fourier analysis
    L = T_edge+T_spont+period   # start of 2nd stimulus cycle
    R = T-T_edge-T_spont        # end of stimulus

    # containers for spike times and rate samples
    spt = Vector{Float64}()                        # holds 1 complete spike train, visible after loop 
    sptime = Vector{Vector{Float64}}(undef, Nrep)  # holds all spike trains, trimmed to stimulus epoch
    sprate = [zeros(Float64, N_trimmed) for _ in 1:Nrep]

    # simulation loop
    for rep in 1:Nrep

        # spiketrain by simulation
        spt = spiketimes(neuron, stimulus, T)

        # extract spikes between start of 2nd cycle and end of stimulus, for Fourier analysis
        sptime[rep] = filter(s-> (s>=L) && (s<=R), spt) .- L
    
        # count spikes
        sc = length(sptime[rep])
        spikecount = spikecount + sc
        #  turn off GLR display if not enough spikes on any rep
        if sc/(Ncycles-1) < minSpikeCount
            show_GLR = false
        end

        # compute GLR in any case
        spr = GLRFt(spt, f, t; rectify = false)       # rate estimate tuned to f

        # rate estimate, edges trimmed
        sprate[rep] = spr[t_index_trimmed]

    end

    # plot GLR firing rate, up to 16 reps & mean
    if GLR_ON  # caller specified
        if show_GLR  # internal switch for when GLR plot gets messy
            for rep in 1:min(Nrep,16)
                lines!(ax1, t_trimmed, sprate[rep], linewidth = 0.5, color = (:salmon1, 1.0))
            end
        end
        lines!(ax1, t_trimmed, mean(sprate), color = :orangered3, linewidth = 2 )
    end

    # plot stimulus, offset by spontaneous mean so as to overly response
    Stlabel = @sprintf "Amplitude = %.2f, Frequency = %.0fHz" A f
    spontmean = 1.0/summarystats[i][1]  # mean rate = 1/mean interval
    stim = spontmean .+ [stimulus(s+T_edge) for s in t_trimmed]

    if SPIKEPLOT_ON
        lines!(ax1, t_trimmed, stim, color = :white, linewidth = 3)
        lines!(ax1, t_trimmed, stim, color = :deepskyblue3, linewidth = show_GLR ? 2 : 0.5, label = Stlabel)
    else
        lines!(ax1, [0], [0], color = :deepskyblue3, label = Stlabel)
    end

    # spike train (2nd rep)
    Slabel = @sprintf "CV = %.3f, CV* = %.3f"  summarystats[i][3] summarystats[i][4]
    splot_lw = (GLR_ON && show_GLR) ? 1.0 : 2.0
    splot!(ax1, filter(s->(s>T_edge) && (s<T-T_edge), spt) .- T_edge, 8.0, splot_lw, label = Slabel)

    # amplitude and phase of spikes between 2nd cycle & end stimulus
    # at frequency f via Fourier coeffs  
    amplitude, phase, dc = fit_sine2spiketrain_Fourier(vcat(sptime...), f, period, Ncycles-1, Nrep)

    #drivenmean = spikecount/(period*(Ncycles-1)*Nrep)

    # draw the fitted response
    gain = amplitude/A
    phaseDeg = phase*180.0/π
    RLabel = @sprintf "Gain = %.2f, Phase = %.1f degrees" gain phaseDeg
    t_cyc = 0.0:glr_dt:(period*(Ncycles-1))   
    if SPIKEPLOT_ON
        FT_linestyle = (GLR_ON && show_GLR) ? :dot : :solid
        lines!((L+(phase/2π)*period-period) .+ t_cyc, dc .+ amplitude*sin.(2π*f*t_cyc), 
                color = :gray0, linewidth = show_GLR ? 2.5 : 1,  linestyle = FT_linestyle, label = RLabel) 
    else
        lines!([0], [0], color=:gray0, label = RLabel)
    end

                # post-process with utility function reposition_legend!()
    # if the legend overlies plot elements
    axislegend(ax1, position = :rt)

    xlims!(0, T_stim+2.0*T_spont)

    ax2 = Axis(F[1,2], aspect  = DataAspect(),
                    title = "Spike phase distribution",
                    xticksvisible = false,
                    yticksvisible = false,
                    xgridvisible = false,
                    ygridvisible = false,
                    xticklabelsvisible = false,
                    yticklabelsvisible = false,
                    leftspinevisible   = false,
                    rightspinevisible  = false,
                    topspinevisible    = false,
                    bottomspinevisible = false
    )
    colsize!(F.layout, 1, Relative(3/4))

                    # titlesize = 10,
    phase_histogram!(ax2, sptime, f, T_spont)

    spikes_cycle = spikecount/((Ncycles-1)*Nrep)
 #   ax2.xlabel =   @sprintf "%d cycles x %d reps, %.2f spikes/cycle, %.2f cycles/spike" Ncycles Nrep 
    ax2.xlabelsize = 12  

    txt1 = @sprintf "%d cycles" Ncycles
    txt2 = @sprintf "%d reps" Nrep
    txt3 = @sprintf "%.4f s/c" spikes_cycle 
    txt4 = @sprintf "%.2f c/s" 1.0/spikes_cycle  

    text!(ax2, 0.7, 0.12, text =txt1,  
            space = :relative,  align = (:left, :top),  fontsize = 14, color = :black)
    text!(ax2, 0.7, 0.06, text =txt2,  
            space = :relative,  align = (:left, :top),  fontsize = 14, color = :black)
    text!(ax2, 0.3, 0.12, text =txt3,  
            space = :relative,  align = (:right, :top),  fontsize = 14, color = :black)
    text!(ax2, 0.3, 0.06, text =txt4,  
            space = :relative,  align = (:right, :top),  fontsize = 14, color = :black)

    display(F)

    return sptime, sprate, t, F

end





