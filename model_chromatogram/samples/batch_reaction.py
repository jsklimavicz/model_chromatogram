import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings


class BatchReaction:
    def __init__(
        self,
        reaction_map,
        initial_concentrations,
        initial_volume,
        additions,
        start_day,
        end_day,
        n_points=1001,
    ) -> None:
        self.initial_concentrations = initial_concentrations
        self.initial_volume = initial_volume
        self.additions = additions
        self.start_day = start_day
        self.end_day = end_day
        self.n_points = n_points
        self.t_span = (start_day, end_day)
        self.t_eval = np.linspace(start_day, end_day, n_points)
        self.t_values = None
        self.state_over_time = None
        self.compound_names = None
        self.reactions = None

        if self.t_values is None or self.state_over_time is None:
            # Setup and solve ODEs
            self._initial_kinetics_setup_with_reactions(reaction_map)
            self._solve_reaction_ODE_with_additions()
        else:
            # Just parse reactions
            self._parse_reactions(reaction_map)

        self.N_over_time = self.state_over_time[:, :-1] * 1e3  # Convert to mmol
        self.V_over_time = self.state_over_time[:, -1]  # Volume over time
        self.C_over_time = (
            self.N_over_time / self.V_over_time[:, np.newaxis]
        )  # Concentrations over time in mmol/L

    def _parse_reactions(self, reaction_map):
        reactions = []
        for reaction in reaction_map:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate_constant = reaction["k"]
            reactions.append((reactants, products, rate_constant))
        return reactions

    def _reaction_rates(self, C, V, reactions, compound_names):
        dNdt = np.zeros(len(compound_names))
        compound_indices = {name: idx for idx, name in enumerate(compound_names)}

        for reactants, products, k in reactions:
            rate = k
            for reactant in reactants:
                rate *= C[compound_indices[reactant]]  # Rate depends on all reactants
            rate_mol_per_time = rate * V  # Convert rate to moles per time

            # Update the moles for reactants and products
            for reactant in reactants:
                dNdt[compound_indices[reactant]] -= rate_mol_per_time
            for product in products:
                dNdt[compound_indices[product]] += rate_mol_per_time

        return dNdt

    def _solve_reaction_ODE_with_additions(self):
        compound_names = self.compound_names
        reactions = self.reactions
        initial_concentrations = self.initial_concentrations
        initial_volume = self.initial_volume
        additions = self.additions
        t_span = self.t_span
        t_eval = self.t_eval

        compound_indices = {name: idx for idx, name in enumerate(compound_names)}
        initial_moles = np.zeros(len(compound_names))
        for compound, conc in initial_concentrations.items():
            idx = compound_indices[compound]
            initial_moles[idx] = conc * initial_volume  # Convert to moles

        initial_state = np.concatenate([initial_moles, [initial_volume]])

        # Collect discrete addition times within t_span
        discrete_times = sorted(
            set(
                [
                    addition["time"]
                    for addition in additions
                    if addition["type"] == "discrete"
                    and t_span[0] < addition["time"] < t_span[1]
                ]
            )
        )

        # Create the list of times where the integration will be split
        split_times = [t_span[0]] + discrete_times + [t_span[1]]

        # Collect solution segments
        t_total = []
        y_total = []

        state = initial_state

        for i in range(len(split_times) - 1):
            t0 = split_times[i]
            t1 = split_times[i + 1]

            # Adjust t_eval to include only times within this interval
            mask = (t_eval >= t0) & (t_eval <= t1)
            t_eval_interval = t_eval[mask]
            if len(t_eval_interval) == 0:
                t_eval_interval = [t0, t1]  # At least include the start and end times

            # Define ODE function for this interval
            def ode_func(t, state):
                N = state[:-1]  # Moles of compounds
                V = state[-1]  # Volume
                C = N / V  # Concentrations

                reaction_dNdt = self._reaction_rates(C, V, reactions, compound_names)
                F_in_N, F_in_V = addition_rates_func(t)
                dNdt = reaction_dNdt + F_in_N
                dVdt = F_in_V  # Volume change due to additions
                return np.concatenate([dNdt, [dVdt]])

            # Define additions function for this interval
            def addition_rates_func(t):
                rates_N = np.zeros(len(compound_names))
                rate_V = 0.0
                for addition in additions:
                    if addition["type"] == "continuous":
                        if addition["start_time"] <= t <= addition["end_time"]:
                            for compound, rate in addition.get("rate", {}).items():
                                idx = compound_indices[compound]
                                rates_N[idx] += rate  # moles per unit time
                            rate_V += addition.get(
                                "volume_rate", 0.0
                            )  # volume per unit time
                return rates_N, rate_V

            sol = solve_ivp(
                ode_func,
                (t0, t1),
                state,
                method="BDF",
                t_eval=t_eval_interval,
                vectorized=False,
            )

            # Append solution
            t_total.extend(sol.t)
            y_total.append(sol.y.T)

            # Update state to last value
            state = sol.y[:, -1]

            # Apply discrete additions at t1 if any
            for addition in additions:
                if addition["type"] == "discrete" and addition["time"] == t1:
                    # Apply the addition once
                    for compound, amount in addition.get("amount", {}).items():
                        idx = compound_indices[compound]
                        state[idx] += amount  # moles added
                    state[-1] += addition.get("volume", 0.0)  # volume added

        # Concatenate all y_total
        y_total = np.concatenate(y_total, axis=0)
        t_total = np.array(t_total)

        # Sort t_total and y_total in case of overlapping intervals
        idx_sort = np.argsort(t_total)
        t_total = t_total[idx_sort]
        y_total = y_total[idx_sort]

        self.t_values = t_total
        self.state_over_time = y_total

    def plot_kinetics_with_reactions(
        self,
        title=None,
        compound_name_mapping=None,
        **kwargs,
    ):

        # Sort the compound names alphabetically and get their indices
        sorted_compound_indices = sorted(
            enumerate(self.compound_names), key=lambda x: x[1]
        )

        plt.figure(figsize=(10, 8))
        for idx, name in sorted_compound_indices:
            conc_list = self.C_over_time[:, idx]
            label = (
                compound_name_mapping.get(name, name) if compound_name_mapping else name
            )
            plt.plot(self.t_values, conc_list, label=label)

        plt.xlabel("Time (days)")
        plt.ylabel("Concentration (mmol/L)")
        if title is not None:
            plt.title(f"{title}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _initial_kinetics_setup_with_reactions(self, reaction_map):
        compound_names = set()
        for reaction in reaction_map:
            compound_names.update(reaction["reactants"] + reaction["products"])
        self.compound_names = list(compound_names)
        self.reactions = self._parse_reactions(reaction_map)

    def get_concentrations_at_times(
        self, time_points, sort=None, asc=True, normalize=None
    ):

        # Create interpolators for each compound
        interpolators = {}
        for idx, name in enumerate(self.compound_names):
            interpolators[name] = interp1d(
                self.t_values,
                self.C_over_time[:, idx],
                kind="linear",
                fill_value="extrapolate",
            )

        # For each time point, create a dict with compound names and concentrations
        concentration_list = []
        for t in time_points:
            concentrations = {}
            for name in self.compound_names:
                conc = interpolators[name](t)
                concentrations[name] = conc

            # Normalization if requested
            if normalize is not None:
                concentrations_copy = concentrations.copy()
                if normalize == "max":
                    max_conc = max(concentrations.values())
                    if not np.isfinite(max_conc) or max_conc == 0:
                        warnings.warn(
                            f"At time {t}, invalid max concentration for normalization."
                        )
                    else:
                        for name in concentrations:
                            concentrations[name] = (
                                concentrations[name] / max_conc
                            ) * 100
                elif normalize == "total":
                    total_conc = sum(concentrations.values())
                    if not np.isfinite(total_conc) or total_conc == 0:
                        warnings.warn(
                            f"At time {t}, invalid total concentration for normalization."
                        )
                    else:
                        for name in concentrations:
                            concentrations[name] = (
                                concentrations[name] / total_conc
                            ) * 100
                elif normalize in self.compound_names:
                    ref_conc = concentrations.get(normalize)
                    if not np.isfinite(ref_conc) or ref_conc == 0:
                        warnings.warn(
                            f"At time {t}, invalid reference concentration for normalization."
                        )
                    else:
                        for name in concentrations:
                            concentrations[name] = (
                                concentrations[name] / ref_conc
                            ) * 100
                else:
                    warnings.warn(
                        f"Invalid normalize option '{normalize}'. Ignoring normalization."
                    )
                    # Do not apply normalization
                    concentrations = concentrations_copy

            # Sorting if requested
            if sort is not None:
                if sort.lower() == "name":
                    concentrations = dict(
                        sorted(
                            concentrations.items(), key=lambda x: x[0], reverse=not asc
                        )
                    )
                elif sort.lower() in ("conc", "concentration"):
                    concentrations = dict(
                        sorted(
                            concentrations.items(), key=lambda x: x[1], reverse=not asc
                        )
                    )
                else:
                    raise ValueError(
                        "Invalid sort option. Use 'name' or 'conc'/'concentration'."
                    )

            concentration_list.append(concentrations)

        return concentration_list
