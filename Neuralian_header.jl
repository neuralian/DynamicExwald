# Neuralian header 
# loads libraries & toolboxes and sets globals
# MGP 2024-25

using Distributions, GLMakie, ImageFiltering, 
    WAV, Sound, PortAudio, SampledSignals, 
    Printf, MLStyle, SpecialFunctions, Random, MAT, 
    BasicInterpolators, DSP, StatsBase,
    Infiltrator, Colors

include("Neuralian_utilities.jl")
include("Neuralian_models.jl")
include("NeuralianFit.jl")
include("NeuralianBayesian.jl")

DEFAULT_SIMULATION_DT = 1.0e-5
PLOT_SIZE = (800, 600)
