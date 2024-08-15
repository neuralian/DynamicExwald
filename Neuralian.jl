# Neuralian Toolbox
# MGP August 2024

# splot:  plot spike train
# GLR: Gaussian local rate filter

using Distributions, Plots, ImageFiltering, Printf

DEFAULT_DT = 1.0e-5
PLOT_SIZE = (800,600)


# plot spiketime vector as spikes
function splot(spiketime::Vector{Float64}, height::Float64=1.0)

    N = length(spiketime)
    plot(ones(2,N).*spiketime', [0.0, height].*ones(2,N), color = :blue, legend = false)
end

# Uniform Gaussian rate filter
# Firing rate estimation by sampling Gaussians centred at spike times
#   spiketimes, sd of Gaussian filter, dt = sampling interval, T = end time
# MGP July 2024
# spiketime in seconds
# sd of Gaussian kernal for each spike (if length(sd)==1 then same kernel for all spikes)
# dt = sample interval in seconds
# pad = pad the start and end of the spike train with mirror image of spikes in first and last pad seconds 
#       (kluge to prevent edge effects. the padded ends are removed before return) (default no pads).
# last sample time (default rounded up to nearest second)
function GLR(spiketime::Vector{Float64}, sd::Vector{Float64}, dt::Float64, pad::Float64 = 0.0, T::Float64=ceil(maximum(spiketime)))

    if length(sd)==1
        sd = sd[]*ones(length(spiketime))
    end

    if length(sd)!=length(spiketime)
        print("length mismatch: length of sd must equal number of spikes")
        return
    end


    print("length(spiketime) = "),     println(length(spiketime))
    print("pad= "),     println(pad)

    if pad > 0.0
        ifront = findall(spiketime[:].<pad)[end:-1:1]  # indices of front pad spikes in reverse order (mirror) 
        iback  = findall(spiketime[:].>(T-pad))[end:-1:1]
        #@infiltrate
        spiketime = pad .+ vcat(2.0*spiketime[1].-spiketime[ifront[1:end-1]], spiketime, 2.0*spiketime[iback[1]] .- spiketime[iback[2:end]])
        sd = vcat(sd[ifront[2:end]], sd, sd[iback[2:end]])
    end
    print("length(spiketime) = "),     println(length(spiketime))

    # vector to hold result
    padN = Int(ceil(pad/dt))    # pad lengths
    sigN = Int(ceil(T)/dt)      # signal length (number of sample points)
    N = sigN+2*padN             # padded signal length
    r = zeros(N)    


    for i in 1:length(spiketime)
        k = Int(round(spiketime[i]/dt)) # ith spike occurs at this sample point
        n = Int(4.0*sd[i]/dt)  # number of sample points within 4 sd each side of spike
        kernel = pdf(Normal(spiketime[i], sd[i]), k*dt .+ (-n:n)*dt) 
        for j in -n:n  # go 4 sd each side
            if (k+j)>=1 && (k+j)<=N 
               r[k+j] += kernel[n+j+1]
            end
        end
    end

  r[padN .+ (1:sigN)]


end

# sample of size N from Wald (Inverse Gaussian) Distribution
# by simulating first passage times of drift-diffusion to barrier (integrate noise to threshold)
# interval (pointer to) vector of intervals (to be computed), must be intially all zeros
# v: drift rate  
# s: diffusion coeff (std deviation of noise)
# a: barrier height 
# dt: default timescale 1.0e-5 = 10 microseconds
function FirstPassageTime_simulate(interval::Vector{Float64}, v::Float64, s::Float64, a::Float64, 
                                   dt::Float64=DEFAULT_DT, T::Float64=1.5*a/v*length(interval))
    
    spike_count = 0
    tick = 0.0
    #@infiltrate
    while spike_count < length(interval) && tick < T  # until run out of space to store intervals or reached end time
        x = 0.0
        t = 0.0
        while(x<a)
            x = x + v*dt + s*randn(1)[]*sqrt(dt)  # integrate noise
            t += dt
            tick += dt
        end
        spike_count = spike_count + 1
        interval[spike_count] = t 
    end
    return (spike_count, tick)   # intervals are in 'interval'
end

# First passage time simulation parameters for specified Wald parameters with threshold (a) = 1
function FirstPassageTime_parameters_from_Wald(mu::Float64, lambda::Float64)

    return (1.0/mu, 1.0/sqrt(lambda), 1.0)  # (v, sigma, alpha) = (drift speed, noise s.d., threshold)

end



# sample of size length(interval) from Wald (Inverse Gaussian) distribution by simulating first passage times
function Wald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, dt::Float64=DEFAULT_DT)

    # drift-diffusion model parameters from Wald parameters
    (v,s,a) = FirstPassageTime_parameters_from_Wald(mu, lambda)
    (spike_count, tick) = FirstPassageTime_simulate(interval, v, s, a, dt)
    if spike_count < length(interval)
        println("Warning: Wald_sample() generated fewer spikes than requested ($spike_count/$N). Increase the length of the interval vector.")
    end
    (v,s,a)
end


# Compare first passage time simulation histogram to InverseGaussian from Distributions.jl 
function test_Wald_sample(mu::Float64, lambda::Float64)

    N = 5000  # sample size
    I = zeros(N)
    (v,s,a) = Wald_sample(I, mu, lambda)
    histogram(I, normalize=:pdf, nbins=100,
               label = @sprintf "First passage times (%.2f, %.2f, %.2f)" v s a)
    t = DEFAULT_DT:DEFAULT_DT:(maximum(I)*1.2)
    plot!(t, pdf(InverseGaussian(mu, lambda), t), 
        label = "Wald ($mu, $lambda)", linewidth = 2.5, legend = :topleft, size = PLOT_SIZE)
    xlims!(0.0, t[end])
end


#function FirstPassageTime_simulate(interval::Vector{Float64}, v::Float64, s::Float64, a::Float64, dt::Float64=DEFAULT_DT, T::Float64=1.5*v/a*length(interval))

# Exponential samples by threshold-crossing in Gaussian noise = Normal(m,s)
function ThresholdTrigger_simulate(interval::Vector{Float64}, m::Float64, s::Float64, a::Float64,  
                                    dt::Float64=DEFAULT_DT, 
                                    T::Float64= 1.5*PoissonTau_from_ThresholdTrigger(m, s, a, dt)*length(interval))

    spike_count = 0
    tick = 0.0
    N = length(interval)
    #@infiltrate
    while spike_count < N  && tick < T
        t = 0.0
        while((m+s*randn()[])<a)
            t += dt
            tick += dt
        end
        spike_count += 1
        interval[spike_count] += t # NB adding so we can use the same interval vector for Exp and Exwald (see Exwaldsim)
    end
    return (spike_count, tick)
end

#function InhomogeneousPoission_simulate(interval::Vector{Float64}, m::Vector{Float64}, s::Float64, dt::Float64=DEFAULT_DT, T::Float64)
#end

# exponential samples by threshold trigger simulation with standard Normal noise
# sample size = length(interval)
function Exponential_sample(interval::Vector{Float64}, tau::Float64, dt::Float64=DEFAULT_DT)

    m = 0.0
    s = 1.0
    threshold = TriggerThreshold_from_PoissonTau(m, s, tau, dt)
    T = 1.5*tau*length(interval)
   # @infiltrate
    ThresholdTrigger_simulate(interval, m, s, threshold, dt)
    (m,s,threshold)  # return trigger mechanism parameters
end

# spike trigger threshold to get mean interval tau between events with noise input Normal(m,s)
function TriggerThreshold_from_PoissonTau(v::Float64, s::Float64, tau::Float64, dt::Float64=DEFAULT_DT)

    quantile(Normal(v,s), 1.0-dt/tau)  # returns threshold (a)

end

# find mean interval length of threshold trigger with input noise Normal(m,s)
function PoissonTau_from_ThresholdTrigger(m::Float64, s::Float64, threshold::Float64,  dt::Float64=DEFAULT_DT)

    dt/(1.0-cdf(Normal(m,s), threshold))

end

# Compare Expsim histogram to Exponential 
function test_Exponential_sample(tau::Float64, dt::Float64=DEFAULT_DT)

    N = 5000  # sample size
    interval = zeros(N)
    (m,s,threshold) = Exponential_sample(interval, tau)
    histogram(interval, normalize=:pdf, nbins=100,
               label = @sprintf "Threshold trigger (%.2f, %.2f, %.2f)" m s threshold)
    t = 0.0:dt:(maximum(interval)*1.2)
    plot!(t, exp.(-t./tau)./tau, linewidth = 2.5, size = PLOT_SIZE,
                label = @sprintf "Exponential ( %.2f)" tau)  #pdf(InverseGaussian(mu, lambda), t))
    xlims!(0.0, t[end])
end

# Exwald samples by simulation
# interval: return vector must be initialized to zeros
#  m:  input noise mean
#  s: input noise sd
# aw: wald threshold (barrier distance)
# ae: exponential trigger threshold 
function Exwald_simulate(interval::Vector{Float64}, v::Float64,  s::Float64, barrier::Float64, trigger::Float64, 
            dt::Float64=DEFAULT_DT, 
            T::Float64=1.5*(barrier/v + PoissonTau_from_ThresholdTrigger(v, s, trigger, dt))*length(interval))

         FirstPassageTime_simulate(interval, v, s, barrier, dt, T)  # inserts Wald intervals 

         ThresholdTrigger_simulate(interval, v, s, trigger, dt, T)  # adds Exponential intervals

end


# Exwald samples by simulation
# sample size = length(interval)
function Exwald_sample(interval::Vector{Float64}, mu::Float64, lambda::Float64, tau::Float64, dt::Float64=DEFAULT_DT)

    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau, dt)            # threshold for Poisson component using same noise

    #@infiltrate
    Exwald_simulate(interval, v, s, barrier, trigger, dt)

end


# Exwald pdf by convolution of Wald and Exponential pdf
function Exwaldpdf(mu::Float64, lambda::Float64, tau::Float64, dt::Float64, T::Float64)

    t = dt:dt:T
    W = pdf(InverseGaussian(mu, lambda), t)
    P = exp.(-t./tau)./tau
    X = imfilter(W, reflect(P))
    X = X/sum(X)/dt  # renormalize (W & P are not normalized because of discrete approx)
    #@infiltrate
end


function test_Exwald_sample(mu, lambda, tau)

    N = 5000  # sample size
    I = zeros(N)
    Exwald_sample(I, mu, lambda, tau)
    (v, s, barrier) = FirstPassageTime_parameters_from_Wald(mu, lambda)   # Drift speed, noise s.d. & barrier height for Wald component
    trigger = TriggerThreshold_from_PoissonTau(v, s, tau)            # threshold for Poisson component using same noise
    histogram(I, normalize=:pdf, nbins=100, size = (800,600), 
               label = @sprintf "Simulation (%.5f, %.5f, %.5f, %.2f)" v s barrier trigger)
   
    dt = 1.0e-5
    T = maximum(I)*1.2
    t = dt:dt:T
    lw = 2.5

    X = Exwaldpdf(mu,lambda, tau, dt, T)
    W = pdf(InverseGaussian(mu, lambda), t)
    P =  exp.(-t./tau)./tau

    plot!(t, W , linewidth = lw, size = PLOT_SIZE, 
               label = @sprintf "Wald (%.5f, %.5f)" mu lambda) 
    plot!(t, P, linewidth = lw, label = @sprintf "Exponential (%.5f)" tau) 
    plot!(t, X, linewidth = lw*1.5, label = @sprintf "Exwald (%.5f, %.5f, %.5f)" mu lambda tau) 
    ylims!(0.0, max(maximum(X), maximum(W))*1.25)
    xlims!(0.0, T)
    #@infiltrate
end