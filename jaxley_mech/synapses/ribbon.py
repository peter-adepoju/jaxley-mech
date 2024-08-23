from typing import Optional

import jax.numpy as jnp
from jaxley.solver_gate import save_exp
from jaxley.synapses.synapse import Synapse

from jaxley_mech.solvers import explicit_euler, newton, rk45, diffrax_implicit



META = {
    "reference": [
        "Schroeder, C., Oesterle, J., Berens, P., Yoshimatsu, T., & Baden, T. (2021). eLife."
    ]
}

class RibbonSynapse(Synapse):
    def __init__(self, name: Optional[str] = None, solver: Optional[str] = "newton"):
        self._name = name = name if name else self.__class__.__name__
        self.solver = solver  # Choose between 'explicit', 'newton', and 'rk45'
        self.synapse_params = {
            f"{name}_gS": 1e-6,  # Maximal synaptic conductance (uS)
            f"{name}_e_syn": 0.0,  # Reversal potential of postsynaptic membrane at the receptor (mV)
            f"{name}_e_max": 1.5,  # Maximum glutamate release
            f"{name}_r_max": 2.0,  # Rate of RP --> IP, movement to the ribbon
            f"{name}_i_max": 4.0,  # Rate of IP --> RRP, movement to the dock
            f"{name}_d_max": 0.1,  # Rate of RP refilling
            f"{name}_RRP_max": 3.0,  # Maximum number of docked vesicles
            f"{name}_IP_max": 10.0,  # Maximum number of vesicles at the ribbon
            f"{name}_RP_max": 25.0,  # Maximum number of vesicles in the reserve pool
            f"{name}_k": 1.0,  # Slope of calcium conversion nonlinearity
            f"{name}_V_half": -35.0,  # Half the voltage that gives maximum glutamate release
        }
        self.synapse_states = {
            f"{name}_exo": 0.75,  # Number of vesicles released
            f"{name}_RRP": 1.5,  # Number of vesicles at the dock
            f"{name}_IP": 5.0,  # Number of vesicles at the ribbon
            f"{name}_RP": 12.5,  # Number of vesicles in the reserve pool
        }
        self.META = META

    def derivatives(self, states, params, pre_voltage):
        """Calculate the derivatives for the Ribbon Synapse system."""
        exo, RRP, IP, RP = states
        e_max, r_max, i_max, d_max, RRP_max, IP_max, RP_max, k, V_half = params

        # Presynaptic voltage to calcium to release probability
        p_d_t = 1.0 / (1.0 + save_exp(-1 * k * (pre_voltage - V_half)))

        # Glutamate release
        e_t = e_max * p_d_t * RRP / RRP_max
        # Rate of RP --> IP, movement to the ribbon
        r_t = r_max * (1 - IP / IP_max) * RP / RP_max
        # Rate of IP --> RRP, movement to the dock
        i_t = i_max * (1 - RRP / RRP_max) * IP / IP_max
        # Rate of RP refilling
        d_t = d_max * exo

        dRP_dt = d_t - r_t
        dIP_dt = r_t - i_t
        dRRP_dt = i_t - e_t
        dExo_dt = e_t - d_t

        return jnp.array([dExo_dt, dRRP_dt, dIP_dt, dRP_dt])

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state using the chosen solver."""
        name = self._name

        # Parameters
        param_tuple = (
            params[f"{name}_e_max"],
            params[f"{name}_r_max"],
            params[f"{name}_i_max"],
            params[f"{name}_d_max"],
            params[f"{name}_RRP_max"],
            params[f"{name}_IP_max"],
            params[f"{name}_RP_max"],
            params[f"{name}_k"],
            params[f"{name}_V_half"],
            pre_voltage,
        )

        # States
        exo, RRP, IP, RP = (
            u[f"{name}_exo"],
            u[f"{name}_RRP"],
            u[f"{name}_IP"],
            u[f"{name}_RP"],
        )
        y0 = jnp.array([exo, RRP, IP, RP])

        # Choose the solver
        if self.solver == "newton":
            y_new = newton(y0, delta_t, self.derivatives, param_tuple[:-1], pre_voltage)

        elif self.solver == "diffrax_implicit":
            y_new = diffrax_implicit(y0, delta_t, self.derivatives, param_tuple)

        elif self.solver == "rk45":
            y_new = rk45(y0, delta_t, self.derivatives, param_tuple[:-1], pre_voltage)

        else:  # Default to explicit Euler
            y_new = explicit_euler(
                y0, delta_t, self.derivatives, param_tuple[:-1], pre_voltage
            )

        new_exo = y_new[0]
        new_RRP = y_new[1]
        new_IP = y_new[2]
        new_RP = y_new[3]
        return {
            f"{name}_exo": new_exo,
            f"{name}_RRP": new_RRP,
            f"{name}_IP": new_IP,
            f"{name}_RP": new_RP,
        }

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        name = self.name
        g_syn = params[f"{name}_gS"] * u[f"{name}_exo"]
        return g_syn * (post_voltage - params[f"{name}_e_syn"])
    

def derivatives_fn(t, y, args):
    """Diffrax form for ODETerm"""
    exo = y[0]
    RRP = y[1]
    IP = y[2]
    RP = y[3]
    e_max, r_max, i_max, d_max, RRP_max, IP_max, RP_max, k, V_half, pre_voltage = args

    # Presynaptic voltage to calcium to release probability
    p_d_t = 1.0 / (1.0 + save_exp(-1 * k * (pre_voltage - V_half)))

    # Glutamate release
    e_t = e_max * p_d_t * RRP / RRP_max
    # Rate of RP --> IP, movement to the ribbon
    r_t = r_max * (1 - IP / IP_max) * RP / RP_max
    # Rate of IP --> RRP, movement to the dock
    i_t = i_max * (1 - RRP / RRP_max) * IP / IP_max
    # Rate of RP refilling
    d_t = d_max * exo

    dRP_dt = d_t - r_t
    dIP_dt = r_t - i_t
    dRRP_dt = i_t - e_t
    dExo_dt = e_t - d_t

    return jnp.array([dExo_dt, dRRP_dt, dIP_dt, dRP_dt])
