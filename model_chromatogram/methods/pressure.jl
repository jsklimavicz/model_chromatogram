module Pressure

export pressure_driver

using DataFrames
include("./viscosity.jl")
using .ViscosityModel

METHANOL = Dict{String,Float64}(
    "density" => 0.792,  # g/cm^3
    "mw" => 32.04,  # g/mol
)
WATER = Dict{String,Float64}(
    "density" => 0.997,  # g/cm^3
    "mw" => 18.01528,  # g/mol
)
ACETONITRILE = Dict{String,Float64}(
    "density" => 0.786,  # g/cm^3
    "mw" => 41.05,  # g/mol
)
THF = Dict{String,Float64}(
    "density" => 0.886,  # g/cm^3
    "mw" => 72.11,  # g/mol
)



struct Column
    length::Float64
    particle_diameter::Float64
    particle_sphericity::Float64
    porosity::Float64
    volume::Float64
    inner_diameter::Float64
end


function calculate_molar_fraction(org_type::String, org_percent::Vector{Float64})::Vector{Float64}
    if org_type == "methanol"
        org_dict = METHANOL
    elseif org_type == "acetonitrile"
        org_dict = ACETONITRILE
    elseif org_type == "thf"
        org_dict = THF
    else
        throw(ArgumentError("Unknown organic type: $org_type"))
    end
    a::Vector{Float64} = org_percent .* org_dict["density"] .* org_dict["mw"]
    b::Vector{Float64} = (100 .- org_percent) .* WATER["density"] .* WATER["mw"]
    return a ./ (a .+ b)
end


function update_solvent_profile!(sp_df::DataFrames.DataFrame, column::Column)
    time_delta::Float64 = sp_df.time[2] - sp_df.time[1]
    sp_df.incremental_CV = sp_df.flow .* time_delta ./ column.volume
    sp_df.incremental_length = sp_df.incremental_CV .* column.length
    sp_df.superficial_velocity = sp_df.flow ./ (column.inner_diameter^2 * π / 400) ./ (100 * 60)
    sp_df.meoh_x = calculate_molar_fraction("methanol", sp_df.percent_meoh)
    sp_df.acn_x = calculate_molar_fraction("acetonitrile", sp_df.percent_acn)
    sp_df.thf_x = calculate_molar_fraction("thf", sp_df.percent_thf)
end

function total_viscosity(
    meoh_x::Vector{Float64},
    acn_x::Vector{Float64},
    thf_x::Vector{Float64},
    t::Vector{Float64},
    p::Float64)::Vector{Float64}

    threshold = 1e-6
    if acn_x + meoh_x + thf_x < threshold
        return viscosity("meoh", t, meoh_x, p)
    end
    water_meoh_x = 1.0 .- (acn_x .+ thf_x)
    meoh_viscosity = viscosity("meoh", t, meoh_x, p)
    acn_viscosity = acn_x > threshold ? viscosity("acn", t, acn_x, p) : 1.0
    thf_viscosity = thf_x > threshold ? viscosity("thf", t, thf_x, p) : 1.0
    return exp.(water_meoh_x .* log.(meoh_viscosity) .+ acn_x .* log.(acn_viscosity) .+ thf_x .* log.(thf_viscosity))
end #total_viscosity

function total_viscosity(
    meoh_x::Float64,
    acn_x::Float64,
    thf_x::Float64,
    t::Float64,
    p::Float64)::Float64

    threshold = 1e-6
    if typeof(meoh_x) == Float64 && acn_x + meoh_x + thf_x < threshold
        return viscosity("meoh", t, meoh_x, p)
    end
    water_meoh_x = 1.0 .- (acn_x .+ thf_x)
    meoh_viscosity = viscosity("meoh", t, meoh_x, p)
    acn_viscosity = acn_x > threshold ? viscosity("acn", t, acn_x, p) : 1.0
    thf_viscosity = thf_x > threshold ? viscosity("thf", t, thf_x, p) : 1.0
    return exp(water_meoh_x * log(meoh_viscosity) + acn_x * log(acn_viscosity) + thf_x * log(thf_viscosity))
end #total_viscosity

function kozeny_carman_model(
    kozeny_carman::Float64,
    v::Vector{Float64},
    η::Vector{Float64},
    l::Vector{Float64})::Vector{Float64}
    pressure = kozeny_carman .* η .* l .* v
    return pressure .* 1e-8
end #kozeny_carman_model

function kozeny_carman_model(
    kozeny_carman::Float64,
    v::Float64,
    η::Float64,
    l::Float64)::Float64
    pressure = kozeny_carman .* η .* l .* v
    return pressure .* 1e-8
end #kozeny_carman_model

function pressure_driver(
    solvent_profile,
    column::AbstractArray;
    column_permeability_factor::Float64=20.0,
    tolerance::Float64=1e-5,
    recalculate_viscosities::Bool=false,
    initial_pressure_guess::Float64=200.0)::Vector{Float64}

    column = Column(column...)
    sp_df::DataFrames.DataFrame = solvent_profile |> DataFrames.DataFrame
    kozeny_carman::Float64 = column_permeability_factor * 150 / (column.particle_diameter * column.particle_sphericity)^2 * (1 - column.porosity)^2 / column.porosity^3
    update_solvent_profile!(sp_df, column)

    viscosities::Vector{Float64} = zeros(length(sp_df.time))
    pressures::Vector{Float64} = zeros(length(sp_df.time))

    for i in 1:length(sp_df.time)
        δ::Float64 = Inf64
        column_fraction_accounted_for::Float64 = 0.0
        lower_index::Int64 = i
        column_unaccounted_for::Float64 = 0
        column_overaccounted_for::Float64 = 0
        while column_fraction_accounted_for < 1
            column_fraction_accounted_for += sp_df.incremental_CV[lower_index]
            if column_fraction_accounted_for >= 1
                column_overaccounted_for = column_fraction_accounted_for - 1
                break
            end
            if lower_index == 1
                column_unaccounted_for = 1 - column_fraction_accounted_for
                break
            end
            lower_index -= 1
        end

        pressure::Float64 = 0.0
        while δ > tolerance
            if recalculate_viscosities
                meoh_x = sp_df.meoh_x[lower_index:i]
                acn_x = sp_df.acn_x[lower_index:i]
                thf_x = sp_df.thf_x[lower_index:i]
                t = sp_df.temperature[lower_index:i]
                p = initial_pressure_guess
                current_η = total_viscosity(meoh_x, acn_x, thf_x, t, p)
            else
                η = total_viscosity(sp_df.meoh_x[i], sp_df.acn_x[i], sp_df.thf_x[i], sp_df.temperature[i], initial_pressure_guess)

                viscosities[i] = η
                current_η = viscosities[lower_index:i]
            end

            incremental_pressures = kozeny_carman_model(kozeny_carman, sp_df.superficial_velocity[lower_index:i], current_η, sp_df.incremental_length[lower_index:i])

            pressure = sum(incremental_pressures)
            if column_unaccounted_for > 0
                pressure += column_unaccounted_for / sp_df.incremental_CV[lower_index] * incremental_pressures[1]
            elseif column_overaccounted_for > 0
                pressure -= column_overaccounted_for / sp_df.incremental_CV[lower_index] * incremental_pressures[end]
            end

            δ = abs(pressure - initial_pressure_guess)
            initial_pressure_guess = pressure
        end #while δ > tolerance
        pressures[i] = pressure

    end

    return pressures

end #pressure_driver

end #module