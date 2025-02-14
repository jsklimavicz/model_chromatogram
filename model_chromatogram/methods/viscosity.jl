# module ViscosityModel

module ViscosityModel

export viscosity, model
using CSV, DataFrames, LsqFit, Statistics

# std: 0.017101766
# resid_sum: 3.1432477
# max_deviation: 0.071045428
MEOH_PARAMS = [-4320.221828656787, -0.04317321464026093, -3.2183386517271124e-5, -4.7505432844620525e-5, 371.38798800124863, 0.0004538706442512652, 0.0007376663726093331, 12.090142949338516, 546.0845649098856, -8.857208509233104, -82.38958460101334, 60.96296363045879, -8.911526703699176, 50.12515565908149, -0.02186510631628006, -110239.53634455631, -0.0016034949067753946, -15261.6589032647, 626.7021731564391]

# std: 0.0073562648
# resid_sum: 3.605563
# max_deviation: 0.023139597
ACN_PARAMS = [1000.5224252994153, 0.014999370610366661, -2.553166508492207e-5, -2.6172687530980885e-6, -515.7737458861527, -0.0077298065190134005, -0.0004490251419335607, 0.001530358383339248, 0.007019136108015722, -0.008560398753365471, 0.0055932598208976924, -2.573000096219173, 2.5699730005382704, -0.9417237250709205, 2.52427453434938e-6, -0.8166156207094488, -2.0742521820349215e-5, 9.338993089732396, 57.90262079548248]

# std:0.012573648
# resid_sum:2.3170631
# max_deviation:0.038478272
THF_PARAMS = [1.6563181482576743, 0.04474692598596613, -8.864950216771027e-5, 4.192451366876064e-6, -1024.3441136864744, -0.0007084160629411966, 5.411585590552736e-5, 0.002836980044460593, -0.3174381545313952, 0.592778406913796, 0.04923437614619724, 20.50005804790994, 41.75880466355388, -39.91674068240163, -6.658731934939264e-9, 119.257853718558, -0.0003326722864187177, -3637.946261739626, 292.4245159089498]

Float_or_Vector = Union{Float64,Vector{Float64}}

"""
    model(data::Matrix{Float64}, params::Vector{Float64})::Vector{Float64}

Given a 3×N matrix `data` with rows [τ, χ, ρ] and a vector of parameters `params`,
returns a vector of predicted viscosities.

The model treats the viscosity as a function of temperature, molar fraction, and pressure using a 3,3 Padé approximant for viscosity as a function of the molar fraction, with some dependence on temperature and pressure, as well as an additive exponential term that depends on temperature and pressure, as well as the molar fraction:

    μ(τ,χ,ρ) = A * exp((B + E*χ + H*χ^2) / (τ - C) + D*τ + (F + I*χ + J*χ^2) * ρ + G*ρ*χ*(τ - C)) + (k0 + (k1 + kt1/(τ - C) + kp1*ρ) * χ + k2*χ^2 + k3*χ^3) / (1 + (k_1 + kt_1/(τ - C) + kp_1*ρ) * χ + k_2*χ^2 + k_3*χ^3)

where the parameters are:

    params = [A, B1, B2, C, D, E, F, G, H, I, k0, k1, k2, k3, kt1, kp1, k_1, k_2, k_3, kt_1, kp_1, J]

"""

function model(data::Matrix{Float64}, params::Vector{Float64})::Vector{Float64}
    τ::Vector{Float64} = data[1, :] .+ 273.15 # temperature in K
    χ::Vector{Float64} = data[2, :]  # molar fraction
    ρ::Vector{Float64} = data[3, :]  # pressure

    B, C, D, E, G, H, I, k0, k1, k2, k3, k_1, k_2, k_3, kp1, kt1, kp_1, kt_1, F = params

    χ2::Vector{Float64} = χ .^ 2
    χ3::Vector{Float64} = χ .^ 3

    temp_exponent = B ./ (τ .- F) .+ C .* τ .+ D .* τ .^ 2
    pressure_exponent = E .* ρ
    mole_fraction_mixing = (G ./ τ .+ H .* τ .+ I .* ρ) .* χ

    ρτ_component::Vector{Float64} = exp.(temp_exponent .+ pressure_exponent .+ mole_fraction_mixing)

    χ_component_num::Vector{Float64} = k0 .+ (k1 .+ kt1 ./ τ .+ kp1 .* ρ) .* χ .+ k2 .* χ2 .+ k3 .* χ3
    χ_component_den::Vector{Float64} = 1 .+ (k_1 .+ kt_1 ./ τ .+ kp_1 .* ρ) .* χ .+ k_2 .* χ2 .+ k_3 .* χ3
    η_list::Vector{Float64} = ρτ_component .* χ_component_num ./ χ_component_den
    return η_list
end

"""
    fit_viscosity(filename::String)

Reads a CSV file containing columns "temp", "x", and "viscosity",
fits the viscosity model, and returns the fitted parameters and the fit object.
"""
function fit_viscosity(filename::String)
    # Read the CSV file into a DataFrame.
    df = CSV.read(filename, DataFrame)

    # Extract the relevant columns. (Make sure these names match your CSV.)
    T::Vector{Float64} = df.temp              # temperature (make sure units are consistent, e.g., Kelvin)
    x::Vector{Float64} = df.x                 # molar fraction (or another appropriate concentration metric)
    p::Vector{Float64} = df.pressure          # pressure 
    μ::Vector{Float64} = df.viscosity         # measured viscosity

    # Arrange the independent variables into a 3×N matrix.
    data::Matrix{Float64} = Matrix([T x p]')

    # Initial guess for parameters.
    if occursin("meoh", filename)
        p0 = MEOH_PARAMS

    elseif occursin("acn", filename)
        p0 = ACN_PARAMS

    elseif occursin("thf", filename)
        p0 = THF_PARAMS
    end

    best = Inf64
    p1 = p0
    best_param = p0
    i = 1
    while i < 100
        try
            fit = curve_fit(model, data, μ, p1)
            comp_val = std(fit.resid)
            if comp_val < best
                best = comp_val
                sort_resids = sort(fit.resid)
                println("index ", i, ": std: ", comp_val, ", min: ", sort_resids[1], ", max: ", sort_resids[end])
                println(fit.param)
                best_param = fit.param
                p0 = p1
                # else
                #     p1 = p0
            end
            i += 1
        catch e
            p1 = best_param
        end
        p1 = p1 .* (1 .+ 0.2 * randn(length(p1)))
    end


    # Perform the curve fit.
    fit = curve_fit(model, data, μ, best_param)

    return fit
end

"""
    viscosity(params, T, x)

Given fitted parameters and values for temperature T, molar fraction χ, and pressure ρ,
returns the predicted viscosity.
"""


function viscosity(solvent_model::String, T::Float64, χ::Float64, ρ::Float64)::Float64
    if occursin("meoh", lowercase(solvent_model))
        p0 = MEOH_PARAMS
    elseif occursin("acn", lowercase(solvent_model))
        p0 = ACN_PARAMS
    elseif occursin("thf", lowercase(solvent_model))
        p0 = THF_PARAMS
    end
    data = Matrix{Float64}([T χ ρ]')
    return model(data, p0)[1]
end

function viscosity(solvent_model::String, T::Vector{Float64}, χ::Vector{Float64}, ρ::Vector{Float64})::Vector{Float64}
    if occursin("meoh", lowercase(solvent_model))
        p0 = MEOH_PARAMS
    elseif occursin("acn", lowercase(solvent_model))
        p0 = ACN_PARAMS
    elseif occursin("thf", lowercase(solvent_model))
        p0 = THF_PARAMS
    end
    data = Matrix{Float64}([T χ ρ]')
    return model(data, p0)
end

end #module