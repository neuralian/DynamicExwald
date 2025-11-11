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
    xneuron = make_Exwald_neuron(EXWparam)

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
    xneuron = make_Exwald_neuron(EXWparam)

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


# deprecated
# Plot ISI distribution for Exwald neuron at specified phases of sinusoidal stimulus
# overlay the model-predicted Exwald for each phase
#   0 <= phase <= 360
# N = number of spikes 
# Exwald_param = (mu, lambda, tau)
# Stim_param = (amplitude, frequency /Schwarz)
# Example call:
#  test_Exwald_Neuron_phasedistribution(500000, xwp, (.25, 1.0), vec([45.0*i for i in 0:7]))
function test_Exwald_Neuron_phasedistribution(N::Int64,
    EXW_param::Tuple{Float64,Float64,Float64},
    Stim_param::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    # extract parameters
    (mu, lambda, tau) = EXW_param
    (A, F) = Stim_param
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron
    spt = Exwald_Neuron_Nspikes(N, (mu, lambda, tau), stimulus)



   

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
    lines!(ax1, t1, [stimulus(t) for t in t1], color = :black, linewidth = 1.0)   # plot 1 cycle of stimulus
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
    fitted_param = zeros(3, NP)   # holds fitted Exwald parameters in columns
    for i in 1:NP

        phaseRadians = -phaseAngle[i] * pi / 180.0


        # intervals at specified phase
        I0 = intervalPhase(spt, phaseAngle[i], F, true)

        ax = (ax[:]..., Axis(Fig[1,i], xlabel = @sprintf "%.0f° " phaseAngle[i] ))
        x0[i] = round(FigRadius - R * cos(phaseRadians) - insetPlotHigh/2.0)       
        y0[i] = round(FigRadius - R * sin(phaseRadians) - insetPlotHigh/2.0)

        T = maximum(I0)
        t = collect(0.0:dt:T)

        # predicted Distributions
        v = v0 + stimulus(phaseAngle[i] / (360.0 * F)) # stimulus at phaseAngle[i]
        (mu_model, lambda_model) = Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        tau_model = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)

        # X = Exwaldpdf(mu_model, lambda_model, tau_model, t)
        # W = pdf(InverseGaussian(mu_model, lambda_model), t)
        # P = exp.(-t ./ tau_model) ./ tau_model

        lw = 1.0
        spi = spi + 1
        H = hist!(I0, bins=128, normalization=:pdf)

        # fit Exwald Model
        (maxf, fitted_param[:,i], ret) = Fit_Exwald_to_ISI(I0, [mu_model, lambda_model, tau_model])

        thisbig = fitted_param[1,i] + fitted_param[3,i]   # mean = mu + tau
        if thisbig > xbig
            xbig = thisbig
        end

        lines!(ax[i], t,  
            Exwaldpdf(fitted_param[1,i], fitted_param[2,i], fitted_param[3,i], t), 
            color = :salmon1, linewidth = 2.0)
        
        # set axis limits and remove ticks
        # display(Fig)
        # xlims!(ax[i], 0.0, 0.2) 
        # ax[i].xticks = vec([-1])
        # ylims!(ax[i], 0.0, ymax)
        # ax[i].yticks = vec([-1])
        
        #text!()
          
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

   Fig

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
    Stim_param::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_SIMULATION_DT)

    # extract parameters
    (mu, lambda, tau) = EXWparam
    (A, F) = Stim_param
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # fractional torsion pendulum with velocity 
    # default bandwidth of fractional approximation is .01-20.0 Hz
    # iniitalized in state (0.0,0.0)
    fpt = make_fractional_Steinhausen_stateUpdate_velocity_fcn(q, 0.0, 0.0)

    # Exwald neuron
    xneuron = make_Exwald_neuron(EXWparam)

    # angular velocity function of time
    w = t -> A*sin(2π*F*t)

    # N Exwald intervals by simulation
    ISI = interspike_intervals(xneuron, t->fpt(w, t), N)

    # spike times 
    spt = cumsum(ISI)

    # rate estimate
    dt4glr = 0.01
    bandwidth = 0.1
    (tr, firingrate) = GLR(spt, [bandwidth], dt4glr);

    Npts = 128
    R0 = 1.0
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

        # predicted Distribution
        v = v0 + w(2π*phaseAngle[i] / 360.0) # stimulus at phaseAngle[i]
        (mu0, lambda0) = Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        tau0 = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)
        if (tau0==Inf) 
            tau0 = .1
        end
   
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
        fittedPDF = Exwaldpdf(fitted_param[1,i], fitted_param[2,i], fitted_param[3,i], t)
        if maximum(fittedPDF) > maxy
            maxy = maximum(fittedPDF)
        end
        lines!(ax[i], t,  fittedPDF, color = :red4, linewidth = 2.5)
        stairs!(bin_centres, H.weights/sum(H.weights*bw), color = :salmon)
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

    (maxf, p, ret) = Fit_Exwald_to_ISI(ISI, [mu, lambda, tau],  [0.1, 0.1, 0.1])

    dt = 1.0e-3
    T = maximum(ISI)
    t = collect(dt:dt:T)
    histogram(ISI, normalize=:pdf)
    plot!(t, Exwaldpdf(p[1], p[2], p[3], t), linewidth=2)
    display(plot!(t, Exwaldpdf(mu, lambda, tau, t), linewidth=2))


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
    afferent = make_Exwald_neuron(EXWparam_1, dt)

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

    return F

end


