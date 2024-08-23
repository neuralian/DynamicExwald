# test Neuralian Toolbox
using Infiltrator

# static (spontaneous) simulation
# test_Wald_sample(.013, 1.0)

# test_Exponential_sample(.001)

# test_Exwald_sample(.013, 0.1, .01)

# dynamic simulation
function inhomogenousPoisson_test(N)

    I = zeros(N)
    # trigger levek for tau = 100ms with N(0,1) input noise
    m0 = 10.0
    trigger = TriggerThreshold_from_PoissonTau(m0, 1.0, 0.1)
    #q(t) = t < 50.0 ? 9.5 : 10.0
    f = 0.25
    a = .25
    q(t) = m0 + a*sin(2*pi*f*t)
    ThresholdTrigger_simulate(I, q, 1.0, trigger)
    spt = cumsum(I)   # spike times from intervals
    bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
    # soundsc(bs, 10000)     # play audio at 10KHz
    gdt = 1.0e-3
    r = GLR(spt,[.5], gdt)
    t = collect((1:length(r)))*gdt
    splot(spt)
    plot!(t, r, size = (1000, 800))
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
    q2(t) = v0 + a*sin(2*pi*f*t)
    FirstPassageTime_simulate(I, q2, s, barrier, dt)
    #@infiltrate
    
    spt = cumsum(I)   # spike times from intervals

    gdt = 1.0e-3
    r = GLR(spt,[.1], gdt)
    t = collect((1:length(r)))*gdt
    splot(spt, 20.0)
    plot!(t, r, size = (1000, 400))
    display(plot!(t, [q2(x) for x in t]))
    bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
    soundsc(bs, 10000)     # play audio at 10KHz
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
    q(t) = v0 + a*sin(2*pi*f*t)

    # Exwald samples by simulating physical model of FPT + Poisson process in series
    Exwald_simulate(I, q, s, barrier, trigger, dt)
    #@infiltrate
    
    # Exwald neuron spike times from Exwald intervals
    spt = cumsum(I)   # spike times from intervals

    # plot spike train
    splot(spt, 20.0)

    # Gaussian rate estimate
    gdt = 1.0e-3                      # sample interval for GLR estimate
    (t,r) = GLR(spt,[.1], gdt)       # rate estimate r at sample times t
   
    # plot rate estimate 
    plot!((t, r), size = (1000, 400))
    # plot stimulus
    display(plot!(t, [q(x) for x in t]))
    
    if listen
        bs = s2b(spt, 1.0e-4)  # binary representation at 10KHz
        soundsc(bs, 10000)     # play audio at 10KHz
    end
end






#     # time vector
#     tvec = dt:dt:maximum(spiketime) 

#     # stimulus 
#     stimulus = [q(x) for x in tvec]

#     return (tvec, stimulus,)
#     #@infiltrate
    
#     # Exwald neuron spike times from Exwald intervals
#     spt = cumsum(I)   # spike times from intervals

#     # plot spike train
#     splot(spt, 20.0)

#     # Gaussian rate estimate
#     gdt = 1.0e-3                      # sample interval for GLR estimate
#     (t,r) = GLR(spt,[.1], gdt)       # rate estimate r at sample times t
   
#     # plot rate estimate 
#     plot!((t, r), size = (1000, 400))
#     # plot stimulus
#     display(plot!(t, [q(x) for x in t]))
    
# end

# test dynamic Exwald with no stimulus
function test_Exwald_Neuron_spontaneous(N::Int64, 
        Exwald_param::Tuple{Float64, Float64, Float64}, dt::Float64 = DEFAULT_DT)

(mu, lambda, tau) = Exwald_param
spt = Exwald_Neuron(N, (mu, lambda, tau), t->0.0)
histogram(diff(spt), bins=64, normalize=:pdf, 

        label = @sprintf "Simulation")
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
ylims!(0.0, 2.0*maximum(X))
title!("Exwald Neuron Model Spontaneous ISI")
#@infiltrate

end

# test dynamic Exwald with sinusoidal stimulus
# stimulus parameters Stim_param = (A, F), A = amplitude, F = frequency (Hz)
function test_Exwald_Neuron_sin(N::Int64, 
    Exwald_param::Tuple{Float64, Float64, Float64}, Stim_param::Tuple{Float64, Float64}, dt::Float64 = DEFAULT_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A*sin(2*π*F*t)

    # simulate Exwald neuron
    spt = Exwald_Neuron(N, (mu, lambda, tau), stimulus)

    # plot spike train
    splot(spt, 20.0, 0.5)

    # Gaussian rate estimate
    gdt = 1.0e-3                      # sample interval for GLR estimate
    (t,r) = GLR(spt,[.1], gdt)       # rate estimate r at sample times t
   
    # plot rate estimate 
    plot!((t, r), size = (1000, 400), linewidth = 2.5)

    # plot spontaneous level
    plot!([t[1], t[end]], v*[1.0, 1.0])
    # plot stimulus
    display(plot!(t, [v + stimulus(x) for x in t]))

    return spt
end

# ISI distribution for Exwald neuron at specified phase of sinusoidal stimulus
#   0 <= phase <= 360
function test_Exwald_Neuron_phasedistribution(N::Int64, 
    Exwald_param::Tuple{Float64, Float64, Float64}, 
    Stim_param::Tuple{Float64, Float64}, 
    phaseAngle::Vector{Float64}, 
    dt::Float64 = DEFAULT_DT)

    # extract parameters
    (mu, lambda, tau) = Exwald_param
    (A, F) = Stim_param
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    # sinusoidal stimulus
    stimulus(t) = A*sin(2*π*F*t)

    # simulate Exwald neuron
    spt = Exwald_Neuron(N, (mu, lambda, tau), stimulus)

    Npts = 128
    R0 = 1.0
    plt = plot(   [R0*sin(2.0*pi*i/Npts) for i in 0:Npts], 
            [R0*cos(2.0*pi*i/Npts) for i in 0:Npts], 
            size=(800,800), xlims = (-2.0, 2.0), ylims = (-2.0, 2.0), 
            bgcolor = :white, gridcolor = :white, showaxis = false)
    annotate!( 0.0, 2.0, ("Exwald Model Neuron ISI Distribution during sinusoidal stimulus",
            14, :center))
    annotate!( 0.0, 1.875, ("Top subplot is in phase with acceleration",
            11, :center))
    insetDiam = 0.18
    R = 0.35 #  plot circle radius relative to width of parent axes
    spi = 1  # subplot index counter
    for i in 1:length(phaseAngle)

    phaseRadians = -phaseAngle[i]*pi/180.0


    # intervals at specified phase
    I0 = intervalPhase(spt, phaseAngle[i], F)

    inset = (1, bbox(R*cos(phaseRadians), R*sin(phaseRadians), 
            insetDiam, insetDiam, :center, :center))
  

    T = maximum(I0)
    t = collect(0.0:dt:T)

    # predicted Distributions
    v_p = v + stimulus(phaseAngle[i]/(360.0*F))
    (mu_p, lambda_p) = Wald_parameters_from_FirstpassageTimeModel(v_p, s, barrier)
    tau_p = PoissonTau_from_ThresholdTrigger(v_p, s, trigger, dt)

    X = Exwaldpdf(mu_p, lambda_p, tau_p, t)
    W = pdf(InverseGaussian(mu_p, lambda_p), t)
    P = exp.(-t ./ tau_p) ./ tau_p

    #@infiltrate

    # y tick label visible at phase 0, x label visible at phase 180
    # if i==3
    #     ytickfontcolor = :black
    # else 
    #     ytickfontcolor = :white
    # end 
    ytickfontcolor = :white
    if i==7
        xtickfontcolor = :black
    else 
        xtickfontcolor = :white
    end    

    lw = 1.0
    spi = spi + 1
    H = histogram!(I0, bins = 64,  
       normalize = :pdf, inset=inset, subplot = i+1, 
       linewidth = 0.1, framestyle = :box, 
       xticks=[.02], xtickfontcolor = xtickfontcolor, xtickdirection = :out,
       yticks = [250.0], ytickfontcolor = ytickfontcolor, ytickdirection = :out)
    plot!(t, W, subplot = i+1)
    plot!(t, P, subplot = i+1)
    plot!(t, X, subplot = i+1, linewidth = 2.0, color = :darkblue, 
            xlims=(0.0, .04), ylims=(0.0, 400.))


   end # phase angles


   # plot cycle 2
   spi = spi + 1
   wavelength = 1.0/F
   (t2, r2) = GLR(spt[findall(spt.<3.0*wavelength)], [.1], dt)# GLR over 3 cycles
   i2 = Int(round(wavelength/dt)):Int(round(2.0*wavelength/dt)) # index 2nd cycle
   r2 = r2[i2]            # extract 2nd cycle
   t2 = (1:length(r2))*dt

   spt2 = spt[findall((spt.>=wavelength) .* (spt.<2*wavelength))] .- wavelength



   centerPlot = plot!(t2, r2, color = :green,  linewidth = 2.0,
      inset = bbox(0.0175, -0.025,  .4, .1, :center, :center), 
      framestyle = :box, xticks = [0.0, 1.0], xlims = (0.0, 1.),
      ylims = (0.0, 100.0), yticks = [50.0], ytickfontcolor = :white,
      subplot = spi)


    splot!(spt2, 20.0, spi)

    #@infiltrate
    plot!(t2, v .+ [30.0*stimulus(t) for t in t2], subplot = spi, 
         linewidth = 2.0, color = :pink)

    annotate!(0.65, 90., ("acceleration", 10, :pink))
    annotate!(0.25, 60., ("GLR", 10, :green))
    # lw = 2.5
    # plot!(t, W, linewidth=lw, size=PLOT_SIZE,
    #     label=@sprintf "Wald (%.5f, %.5f)" mu_p lambda_p)
    # plot!(t, P, linewidth=lw, label=@sprintf "Exponential (%.5f)" tau_p)
    # plot!(t, X, linewidth=lw * 1.5, label=@sprintf "Exwald (%.5f, %.5f, %.5f)" mu_p lambda_p tau_p)
    # ylims!(0.0, max(maximum(X), maximum(W)) * 1.25)
    # xlims!(0.0, T)
    # ylims!(0.0, 2.0*maximum(X))
    # display(title!("Exwald Neuron Model ISI at phase angle $phaseAngle"))

    display(plt)

    # png("nameThisFigure")
    
    spt

end