# test Neuralian Toolbox
using Infiltrator

# static (spontaneous) simulation
# test_Wald_sample(.013, 1.0)

# test_Exponential_sample(.001)

# test_Exwald_sample(.013, 0.1, .01)

# Compare first passage time simulation histogram to InverseGaussian from Distributions.jl 
function test_Wald_sample(mu::Float64, lambda::Float64)

    N = 5000  # sample size
    interval = zeros(N)
    Wald_sample(interval, mu, lambda)
    histogram(interval, normalize=:pdf, nbins=100,
        label=@sprintf "First passage times (%.3f, %.3f)" mu lambda)
    t = DEFAULT_DT:DEFAULT_DT:(maximum(interval)*1.2)
    plot!(t, pdf(InverseGaussian(mu, lambda), t),
        label="Wald ($mu, $lambda)", linewidth=2.5, legend=:topleft, size=PLOT_SIZE)
    xlims!(0.0, t[end])
end

# Compare Expsim histogram to Exponential 
function test_Exponential_sample(tau::Float64, dt::Float64=DEFAULT_DT)

    N = 5000  # sample size
    interval = zeros(N)
    (m, s, threshold) = Exponential_sample(interval, tau)
    histogram(interval, normalize=:pdf, nbins=100,
        label=@sprintf "Threshold trigger (%.2f, %.2f, %.2f)" m s threshold)
    t = 0.0:dt:(maximum(interval)*1.2)
    plot!(t, exp.(-t ./ tau) ./ tau, linewidth=2.5, size=PLOT_SIZE,
        label=@sprintf "Exponential ( %.2f)" tau)  #pdf(InverseGaussian(mu, lambda), t))
    xlims!(0.0, t[end])
end


# dynamic simulation
function inhomogenousPoisson_test(N)

    I = zeros(N)
    # trigger levek for tau = 100ms with N(0,1) input noise
    m0 = 10.0
    trigger = TriggerThreshold_from_PoissonTau(m0, 1.0, 0.1)
    #q(t) = t < 50.0 ? 9.5 : 10.0
    f = 0.25
    a = 0.25
    q(t) = m0 + a * sin(2 * pi * f * t)
    ThresholdTrigger_simulate(I, q, 1.0, trigger)
    spt = cumsum(I)   # spike times from intervals
    bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
    # soundsc(bs, 10000)     # play audio at 10KHz
    gdt = 1.0e-3
    r = GLR(spt, [0.5], gdt)
    t = collect((1:length(r))) * gdt
    splot(spt)
    plot!(t, r, size=(1000, 800))
    plot!(t, [q(x) for x in t])

end

# dynamic simulation
function inhomogenousWald_test(N)

    I = zeros(N)
    # FPT process parameters for spontaneous Exwald
    mu = 0.013
    lambda = 0.1
    dt = DEFAULT_DT
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component

    #q(t) = t < 50.0 ? 9.5 : 10.0
    f = 1.0
    a = 20.0
    q2(t) = v0 + a * sin(2 * pi * f * t)
    FirstPassageTime_simulate(I, q2, s, barrier, dt)
    #@infiltrate

    spt = cumsum(I)   # spike times from intervals

    gdt = 1.0e-3
    r = GLR(spt, [0.1], gdt)
    t = collect((1:length(r))) * gdt
    splot(spt, 20.0)
    plot!(t, r, size=(1000, 400))
    display(plot!(t, [q2(x) for x in t]))
    bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
    soundsc(bs, 10000)     # play audio at 10KHz
end



function test_Exwald_sample(mu, lambda, tau)

    N = 5000  # sample size
    I = zeros(N)
    Exwald_sample(I, mu, lambda, tau)
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau)            # threshold for Poisson component using same noise
    histogram(I, normalize=:pdf, nbins=100, size=(800, 600),
        label=@sprintf "Simulation (%.5f, %.5f, %.5f, %.2f)" v s barrier trigger)

    dt = 1.0e-5
    T = maximum(I) * 1.2
    t = collect(dt:dt:T)
    lw = 2.5

    X = Exwaldpdf(mu, lambda, tau, t)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    plot!(t, W, linewidth=lw, size=PLOT_SIZE,
        label=@sprintf "Wald (%.5f, %.5f)" mu lambda)
    plot!(t, P, linewidth=lw, label=@sprintf "Exponential (%.5f)" tau)
    plot!(t, X, linewidth=lw * 1.5, label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)
    ylims!(0.0, max(maximum(X), maximum(W)) * 1.25)
    xlims!(0.0, T)
    #@infiltrate
end


function dynamicExwald_test(N, listen::Bool=false)

    I = zeros(N)    # generate sample of size N 

    # baseline (spontaneous) Exwald parameters
    mu = 0.013
    lambda = 0.1
    tau = 0.01

    dt = DEFAULT_DT  # just to be clear

    # First passage time model parameters for spontaneous Wald component 
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)

    # trigger threshold for spontaneous (mean==tau) Exponwential samples  with N(v0,s) noise
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)

    # sinusoidaal stimulus (input noise mean) parameters
    f = 1.0         # modulation frequency Hz (nb not spike frequency/firing rate)
    a = 0.0        # modulation amplitude

    # stimulus waveform
    q(t) = v0 + a * sin(2 * pi * f * t)

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    Exwald_simulate(I, q, s, barrier, trigger, dt)
    #@infiltrate

    # Exwald neuron spike times from Exwald intervals
    spt = cumsum(I)   # spike times from intervals

    # plot spike train
    splot(spt, 20.0)

    # Gaussian rate estimate
    gdt = 1.0e-3                      # sample interval for GLR estimate
    (t, r) = GLR(spt, [0.1], gdt)       # rate estimate r at sample times t

    # plot rate estimate 
    plot!((t, r), size=(1000, 400))
    # plot stimulus
    display(plot!(t, [q(x) for x in t]))

    if listen
        bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
        soundsc(bs, 10000)     # play audio at 10KHz
    end
end



# test dynamic Exwald with no stimulus
function test_Exwald_Neuron_spontaneous(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64}, dt::Float64=DEFAULT_DT)

    (mu, lambda, tau) = Exwald_param
    spt = Exwald_Neuron(N, (mu, lambda, tau), t -> 0.0)
    histogram(diff(spt), bins=64, normalize=:pdf, label=@sprintf "Simulation")
    xlims!(0.0, xlims()[2])

    T = maximum(diff(spt))
    t = collect(0.0:dt:T)

    X = Exwaldpdf(mu, lambda, tau, t)
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t ./ tau) ./ tau

    lw = 2.5
    plot!(t, W, linewidth=lw, size=PLOT_SIZE,
        label=@sprintf "Wald (%.5f, %.5f)" mu lambda)
    plot!(t, P, linewidth=lw, label=@sprintf "Exponential (%.5f)" tau)
    plot!(t, X, linewidth=lw * 1.5, label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau)
    ylims!(0.0, max(maximum(X), maximum(W)) * 1.25)
    xlims!(0.0, T)
    ylims!(0.0, 2.0 * maximum(X))
    title!("Exwald Neuron Model Spontaneous ISI")
    #@infiltrate

end

# test dynamic Exwald with sinusoidal stimulus
# stimulus parameters Stim_param = (A, F), A = amplitude, F = frequency (Hz)
function test_Exwald_Neuron_sin(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64}, Stim_param::Tuple{Float64,Float64}, dt::Float64=DEFAULT_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron
    spt = Exwald_Neuron(N, (mu, lambda, tau), stimulus)

    # plot spike train
    splot(spt, 20.0, 0.5)

    # Gaussian rate estimate
    gdt = 1.0e-3                      # sample interval for GLR estimate
    (t, r) = GLR(spt, [0.1], gdt)       # rate estimate r at sample times t

    # plot rate estimate 
    plot!((t, r), size=(1000, 400), linewidth=2.5)

    # plot spontaneous level
    plot!([t[1], t[end]], v * [1.0, 1.0])
    # plot stimulus
    display(plot!(t, [v + stimulus(x) for x in t]))

    return spt
end

# Plot ISI distribution for Exwald neuron at specified phases of sinusoidal stimulus
# overlay the model-predicted Exwald for each phase
#   0 <= phase <= 360
function test_Exwald_Neuron_phasedistribution(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64},
    Stim_param::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron
    spt = Exwald_Neuron(N, (mu, lambda, tau), stimulus)

    Npts = 128
    R0 = 1.0
    plt = plot([R0 * sin(2.0 * pi * i / Npts) for i in 0:Npts],
        [R0 * cos(2.0 * pi * i / Npts) for i in 0:Npts],
        size=(800, 800), xlims=(-2.0, 2.0), ylims=(-2.0, 2.0),
        bgcolor=:white, gridcolor=:white, showaxis=false, legend = false)
    annotate!(0.0, 2.0, ("Exwald Model Neuron ISI Distribution during sinusoidal stimulus",
        14, :center))
    annotate!(0.0, 1.875, ("Top subplot is in phase with acceleration",
        11, :center))
    insetDiam = 0.18
    R = 0.35 #  plot circle radius relative to width of parent axes
    spi = 1  # subplot index counter
    for i in 1:length(phaseAngle)

        phaseRadians = -phaseAngle[i] * pi / 180.0


        # intervals at specified phase
        I0 = intervalPhase(spt, phaseAngle[i], F)

        inset = (1, bbox(R * cos(phaseRadians), R * sin(phaseRadians),
            insetDiam, insetDiam, :center, :center))


        T = maximum(I0)
        t = collect(0.0:dt:T)

        # predicted Distributions
        v_p = v + stimulus(phaseAngle[i] / (360.0 * F))
        (mu_model, lambda_model) = Wald_parameters_from_FirstpassageTimeModel(v_p, s, barrier)
        tau_model = PoissonTau_from_ThresholdTrigger(v_p, s, trigger, dt)

        X = Exwaldpdf(mu_model, lambda_model, tau_model, t)
        W = pdf(InverseGaussian(mu_model, lambda_model), t)
        P = exp.(-t ./ tau_model) ./ tau_model

        #@infiltrate

        # y tick label visible at phase 0, x label visible at phase 180
        # if i==3
        #     ytickfontcolor = :black
        # else 
        #     ytickfontcolor = :white
        # end 
        ytickfontcolor = :white
        if i == 7
            xtickfontcolor = :black
        else
            xtickfontcolor = :white
        end

        lw = 1.0
        spi = spi + 1
        H = histogram!(I0, bins=64,
            normalize=:pdf, inset=inset, subplot=i + 1,
            linewidth=0.1, framestyle=:box,
            xticks=[0.02], xtickfontcolor=xtickfontcolor, xtickdirection=:out,
            yticks=[250.0], ytickfontcolor=ytickfontcolor, ytickdirection=:out)
        plot!(t, W, subplot=i + 1)
        plot!(t, P, subplot=i + 1)
        plot!(t, X, subplot=i + 1, linewidth=2.0, color=:darkblue, legend = false, 
            xlims=(0.0, 0.04), ylims=(0.0, 400.0))


    end # phase angles


    # plot cycle 2
    spi = spi + 1
    wavelength = 1.0 / F
    (t2, r2) = GLR(spt[findall(spt .< 3.0 * wavelength)], [0.1], dt)# GLR over 3 cycles
    i2 = Int(round(wavelength / dt)):Int(round(2.0 * wavelength / dt)) # index 2nd cycle
    r2 = r2[i2]            # extract 2nd cycle
    t2 = (1:length(r2)) * dt

    spt2 = spt[findall((spt .>= wavelength) .* (spt .< 2 * wavelength))] .- wavelength



    centerPlot = plot!(t2, r2, color=:green, linewidth=2.0,
        inset=bbox(0.0175, -0.025, 0.4, 0.1, :center, :center),
        framestyle=:box, xticks=[0.0, 1.0], xlims=(0.0, 1.0),
        ylims=(0.0, 100.0), yticks=[50.0], ytickfontcolor=:white,
        subplot=spi)


    splot!(spt2, 20.0, spi)

    #@infiltrate
    plot!(t2, v .+ [30.0 * stimulus(t) for t in t2], subplot=spi,
        linewidth=2.0, color=:pink)

    annotate!(0.65, 90.0, ("acceleration", 10, :pink))
    annotate!(0.25, 60.0, ("GLR", 10, :green))
    # lw = 2.5
    # plot!(t, W, linewidth=lw, size=PLOT_SIZE,
    #     label=@sprintf "Wald (%.5f, %.5f)" mu_p lambda_model)
    # plot!(t, P, linewidth=lw, label=@sprintf "Exponential (%.5f)" tau_model)
    # plot!(t, X, linewidth=lw * 1.5, label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu_p lambda_model tau_model)
    # ylims!(0.0, max(maximum(X), maximum(W)) * 1.25)
    # xlims!(0.0, T)
    # ylims!(0.0, 2.0*maximum(X))
    # display(title!("Exwald Neuron Model ISI at phase angle $phaseAngle"))

    display(plt)

    # png("nameThisFigure")

    spt

end


# test fitting Exwald model to model-generated ISI data
# N = sample size
function test_fit_Exwald_to_ISI(N::Int64, mu::Float64, lambda::Float64, tau::Base.Float64)

    # simulate N spontaneous spikes (= N intervals, starting from t==0)
    #ISI = Exwald_Neuron( N, (mu, lambda, tau), t -> 0.0, DEFAULT_DT, true )

    ISI = Exwald_sample_sum(N, mu, lambda, tau)

    (maxf, p, ret) = Fit_Exwald_to_ISI(ISI, [0.5, 0.5, 0.5])

    dt = 1.0e-3
    T = maximum(ISI)
    t = collect(dt:dt:T)
    histogram(ISI, normalize=:pdf)
    plot!(t, Exwaldpdf(p[1], p[2], p[3], t), linewidth=2)
    display(plot!(t, Exwaldpdf(mu, lambda, tau, t), linewidth=2))

    p

end


# Extract ISI distribution for Exwald model neuron at specified phases of sinusoidal stimulus
# fit Exwald models to ISI data
# Plot estimated Exwald parameters vs (known) model parameters at each phase
#   0 <= phase <= 360
# eg call  test_fit_Exwald_Neuron_phasedistribution(50000, (.013, .1, .01), (1.0, 2.0), collect(0.0:45.0:360.));
function test_fit_Exwald_Neuron_phasedistribution(N::Int64,
    Exwald_param::Tuple{Float64,Float64,Float64},
    Stim_param::Tuple{Float64,Float64},
    phaseAngle::Vector{Float64},
    dt::Float64=DEFAULT_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v0, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v0, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A * sin(2 * π * F * t)

    # simulate Exwald neuron
    spt = Exwald_Neuron(N, (mu, lambda, tau), stimulus)

    Nphases = length(phaseAngle)

    # layout plot window
    # top row for histograms, bottom row for parameter fits
    lyt = @layout [
        grid(1,Nphases){.1h}
        grid(3,1) 
    ]
    
    Npts = 128
    R0 = 1.0
    plt = plot(layout = lyt,
        size=(800, 800), bgcolor=:white, gridcolor=:white, showaxis=false)



#     @infiltrate


    R = 0.35 #  plot circle radius relative to width of parent axes
    spi = 1  # subplot index counter


    # Extract interval lengths at specified phases of stimulus

    y_inset_offset = 0.05
    x_inset_offset = 0.05
    inset_h = (1.0-2.0*y_inset_offset)/Nphases
    inset_w = 0.2

    # arrays to hold fitted parameters
    mu_hat = zeros(Nphases)
    lambda_hat = zeros(Nphases)
    tau_hat = zeros(Nphases)

    for i in 1:Nphases

        phaseRadians = -phaseAngle[i] * pi / 180.0


        # intervals at ith phase angle
        ISI_i = intervalPhase(spt, phaseAngle[i], F)

        # fit model
        (maxf, p, ret) = Fit_Exwald_to_ISI(ISI_i, [0.5, 0.5, 0.5])

        mu_hat[i]     = p[1]
        lambda_hat[i] = p[2]
        tau_hat[i]    = p[3]

        # drift parameter of physical model
        v = v0 + stimulus(phaseAngle[i] / (360.0 * F))   # drift speed is spontaneous + stimulus-driven

        # Exwald model parameters from physical model parameters
        (mu_model, lambda_model) = Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        tau_model = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)

        # timescale for model evaluation (time delay from last spike)
        Td = 5.0*(mu+tau)
        td = collect(0.0:dt:Td)
        
        # evaluate "true" Exwald model pdf and its Ex- and -Wald components
        Exwald_model   = Exwaldpdf(mu_model, lambda_model, tau_model, td)
        Wald_model     = pdf(InverseGaussian(mu_model, lambda_model), td)
        Poisson_model  = exp.(-td ./ tau_model) ./ tau_model

        # fitted model
        Exwald_fitted = Exwaldpdf(mu_hat[i], lambda_hat[i],  tau_hat[i], td)

        #@infiltrate

        # specify inset subplot within plt for drawing ith ISI histogram + fitted and theoretical model
       # inset = (1, bbox(x_inset_offset, y_inset_offset + (i-1)*inset_h, inset_w, 0.8*inset_h, :bottom, :left))




        # y tick label visible at phase 0, x label visible at phase 180
        # if i==3
        #     ytickfontcolor = :black
        # else 
        #     ytickfontcolor = :white
        # end 
        ytickfontcolor = :white
        if i == 7
            xtickfontcolor = :black
        else
            xtickfontcolor = :white
        end
#title = @sprintf "%.0f" 30.0*i, titlecolor = :black, 
        lw = 2.0
        spi = spi + 1
        H = histogram!(ISI_i, bins=0:1.0e-3:Td,
            normalize=:pdf, subplot=i, 
            linewidth=0.1, framestyle=:box,
            xticks=[0.02], xtickfontcolor=xtickfontcolor, xtickdirection=:out,
            yticks=[250.0], ytickfontcolor=ytickfontcolor, ytickdirection=:out)
        plot!(td, Wald_model, color = :grey, subplot=i)
        plot!(td, Poisson_model, color = :grey, subplot=i)
        plot!(td, Exwald_model, subplot=i, linewidth=lw, color=:darkorange,
            xlims=(0.0, Td), ylims=(0.0, 120.0), legend = :false)
        plot!(td, Exwald_fitted, subplot=i, linewidth=lw, color=:crimson)   
        mytext =  @sprintf "%2.0f°" phaseAngle[i]
        annotate!(0.05, 60.0, 
            text(mytext, 10), subplot = i)     
       # annotate!(0.03, 80.0, text("Hello")      )


    end # phase angles

    # caLculate mu, lambda and tau over stimulus cycle
    dθ = 1.0  # 5° steps
    θ = 0.0:dθ:360.0
    Ns = length(θ)
    mu_dynamic     = zeros(Ns)
    lambda_dynamic = zeros(Ns)
    tau_dynamic    = zeros(Ns)

    @infiltrate
    for i in 1:Ns 
        v = v0 + stimulus( θ[i]/(360.0*F))  
        (mu_dynamic[i], lambda_dynamic[i]) = 
             Wald_parameters_from_FirstpassageTimeModel(v, s, barrier)
        tau_dynamic[i] = PoissonTau_from_ThresholdTrigger(v, s, trigger, dt)
    end


    #@infiltrate

    # plot model mu during stimulus
    ymin = minimum(mu_dynamic)
    ymax = maximum(mu_dynamic)
    yrange = ymax - ymin
    plot!(θ, mu_dynamic, 
        ylims=(ymin - 0.25*yrange, ymax + 0.25*yrange), framestyle=:box, legend = false, 
        showaxis = true, subplot = Nphases + 1)
    ymin = minimum(lambda_dynamic)
    ymax = maximum(lambda_dynamic)
    yrange = ymax - ymin
    plot!(θ, lambda_dynamic,  
        ylims=(ymin - 0.25*yrange, ymax + 0.25*yrange), framestyle=:box, legend = false, 
        showaxis = true, subplot = Nphases + 2)
    ymin = minimum(tau_dynamic)
    ymax = maximum(tau_dynamic)
    yrange = ymax - ymin    
    plot!(θ, tau_dynamic,  
        ylims=(ymin - 0.25*yrange, ymax + 0.25*yrange), legend = false, 
        showaxis = true, framestyle=:box, subplot = Nphases + 3)    




    display(plt)

    # png("nameThisFigure")

    (mu_hat, lambda_hat, tau_hat, spt)

end
