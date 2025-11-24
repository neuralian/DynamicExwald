# Neuralian header 
# loads libraries & toolboxes and sets globals
# MGP 2024-25

using Distributions, CairoMakie, ImageFiltering, 
    WAV, Sound, PortAudio, SampledSignals, LinearAlgebra,
    Printf, MLStyle, SpecialFunctions, Random, MAT, 
    BasicInterpolators, DSP, StatsBase, JLD2,
    Infiltrator, Revise, Colors, Format

include("Neuralian_utilities.jl")
include("Neuralian_models.jl")
include("NeuralianFit.jl")
include("NeuralianBayesian.jl")

DEFAULT_SIMULATION_DT = 5.0e-6
PLOT_SIZE = (800, 600)
