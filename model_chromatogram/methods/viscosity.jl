# module ViscosityModel

module ViscosityModel

export viscosity, model

using CSV, DataFrames, LsqFit, Statistics

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

    A, B, C, D, E, F, G, H, I, k0, k1, k2, k3, kt1, kp1, k_1, k_2, k_3, kt_1, kp_1, J = params

    # Here we assume the model: μ(τ,χ,ρ) = A * exp(B1/τ + B2*τ + C*ρ + D*(ρ/τ)) * (∑_{i=-1}^5 k_i * χ^i)
    χ2::Vector{Float64} = χ .^ 2
    χ3::Vector{Float64} = χ .^ 3
    τp::Vector{Float64} = τ .- C

    ρτ_component::Vector{Float64} = A .* exp.((B .+ E .* χ .+ H .* χ2) ./ τp .+ D .* τ .+ (F .+ I .* χ .+ J .* χ2) .* ρ .+ G .* ρ .* χ .* τp)
    # ρτ_component::Vector{Float64} = A .* exp.(B ./ τp)
    χ_component_num::Vector{Float64} = k0 .+ (k1 .+ kt1 ./ τp .+ kp1 .* ρ) .* χ .+ k2 .* χ2 .+ k3 .* χ3
    χ_component_den::Vector{Float64} = 1 .+ (k_1 .+ kt_1 ./ τp .+ kp_1 .* ρ) .* χ .+ k_2 .* χ2 .+ k_3 .* χ3
    η_list::Vector{Float64} = ρτ_component .+ χ_component_num ./ χ_component_den
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
        # std: 0.01730275310410752, min: -0.059155810056926494, max: 0.0520604709066288
        p0 = [784418.1567955717, -216.8026354754598, 142.16031890839787, -0.04240832727701839, 244.37201248599547, -9.753805625699765e-5, 8.107244920614684e-6, -369.6331176919233, 0.00013562337206193137, 0.28731847351056145, -10.78307989279897, 15.673187316615138, -28.17191398504816, 130.90903000953827, 8.982483009082834e-5, -56.498223214835285, 84.6845719378435, -135.8697786214959, 4836.486656657206, 0.000740405697041416, -0.0005685792018373721]

    elseif occursin("acn", filename)
        # std: 0.00745909360361934, min: -0.028134649365006537, max: 0.023999568245409497
        p0 = [0.014956786275833366, 971.3448272624897, 14.680014217543711, 0.0053753007446537, 191.07308814476835, -3.7621990858728146e-6, -2.0674146418955022e-6, -406.67145357423016, 0.0012900005749072185, -1.3857672598727158, -1.490584271496211, -4.981676597685795, 3.8943327834878265, 763.2452955464573, 8.900999221324344e-5, 1.5938079547512904, 3.277533246291744, -1.1744119388861418, -836.1253002501501, -0.000495402949715956, -0.0003739239266524791]

    elseif occursin("thf", filename)
        # std: 0.013402694326147225, min: -0.04604680646372605, max: 0.031064509566325782
        p0 = [-0.07552431263388727, -1.4988592107965513, 293.0490243489337, 0.008120770616777049, -120.42126965266883, 3.03809290443251e-7, -3.922551338896851e-6, 1.9877231772471573, 9.63359535195254e-5, 1.5161726357306675, 2.2826365273244758, -3.6474465191850025, 1.9263734251673486, -1.1322851249256485, 7.94050146911824e-5, 0.8483444385736404, 8.718891210768659, -5.18086534227937, -9.867978774119825, -0.0001316531911832307, 0.00010671314096280391]
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

# std: 0.01730275310410752, min: -0.059155810056926494, max: 0.0520604709066288
MEOH_PARAMS = [784418.1567955717, -216.8026354754598, 142.16031890839787, -0.04240832727701839, 244.37201248599547, -9.753805625699765e-5, 8.107244920614684e-6, -369.6331176919233, 0.00013562337206193137, 0.28731847351056145, -10.78307989279897, 15.673187316615138, -28.17191398504816, 130.90903000953827, 8.982483009082834e-5, -56.498223214835285, 84.6845719378435, -135.8697786214959, 4836.486656657206, 0.000740405697041416, -0.0005685792018373721]

# std: 0.00745909360361934, min: -0.028134649365006537, max: 0.023999568245409497
ACN_PARAMS = [0.014956786275833366, 971.3448272624897, 14.680014217543711, 0.0053753007446537, 191.07308814476835, -3.7621990858728146e-6, -2.0674146418955022e-6, -406.67145357423016, 0.0012900005749072185, -1.3857672598727158, -1.490584271496211, -4.981676597685795, 3.8943327834878265, 763.2452955464573, 8.900999221324344e-5, 1.5938079547512904, 3.277533246291744, -1.1744119388861418, -836.1253002501501, -0.000495402949715956, -0.0003739239266524791]

# std: 0.013402694326147225, min: -0.04604680646372605, max: 0.031064509566325782
THF_PARAMS = [-0.07552431263388727, -1.4988592107965513, 293.0490243489337, 0.008120770616777049, -120.42126965266883, 3.03809290443251e-7, -3.922551338896851e-6, 1.9877231772471573, 9.63359535195254e-5, 1.5161726357306675, 2.2826365273244758, -3.6474465191850025, 1.9263734251673486, -1.1322851249256485, 7.94050146911824e-5, 0.8483444385736404, 8.718891210768659, -5.18086534227937, -9.867978774119825, -0.0001316531911832307, 0.00010671314096280391]


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