# solvers.py

from typing import Any, Callable, Optional, Tuple, Dict
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from diffrax import ImplicitEuler, ODETerm, diffeqsolve
from jax.scipy.linalg import solve


### SolverExtension base class

class SolverExtension:
    def __init__(
        self,
        solver: Optional[str] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.solver_name = solver
        self.solver_args = {"rtol": rtol, "atol": atol, "max_steps": max_steps}

        if solver is None:
            raise ValueError(
                "Solver must be specified (`newton`, `explicit`, `rk45`, "
                "`diffrax_implicit` or `gillespie`)."
            )
        elif solver == "diffrax_implicit":
            # prepare diffrax implicit Euler
            self.term = ODETerm(self.derivatives)
            root_finder = optx.Newton(rtol=rtol, atol=atol)
            self.diffrax_solver = ImplicitEuler(root_finder=root_finder)

        # dispatch to the chosen solver
        self.solver_func = self._get_solver_func(solver)

    def __getstate__(self) -> Dict[str, Any]:
        # store solver name instead of function reference
        state = self.__dict__.copy()
        state["solver_func"] = self.solver_name
        return state

    def __setstate__(self, state: Dict[str, Any]):
        # restore and rebind solver_func
        self.__dict__.update(state)
        self.solver_func = self._get_solver_func(state["solver_func"])

    def _get_solver_func(self, solver: str) -> Callable:
        solvers = {
            "newton":             self._newton_wrapper,
            "explicit":           explicit_euler,
            "rk45":               rk45,
            "diffrax_implicit":   self._diffrax_implicit_wrapper,
            "gillespie":          self._gillespie_wrapper,
        }
        if solver not in solvers:
            raise ValueError(
                f"Solver {solver!r} not recognized. Supported: {list(solvers)}"
            )
        return solvers[solver]

    def _newton_wrapper(
        self,
        y0: jnp.ndarray,
        dt: float,
        derivatives_func: Callable[..., jnp.ndarray],
        args: Tuple[Any, ...],
    ) -> jnp.ndarray:
        return newton(
            y0,
            dt,
            derivatives_func,
            *args,
            rtol=self.solver_args["rtol"],
            atol=self.solver_args["atol"],
            max_steps=self.solver_args["max_steps"],
        )

    def _diffrax_implicit_wrapper(
        self,
        y0: jnp.ndarray,
        dt: float,
        derivatives_func: Callable[..., jnp.ndarray],
        args: Tuple[Any, ...],
    ) -> jnp.ndarray:
        return diffrax_implicit(
            y0=y0,
            dt=dt,
            derivatives_func=derivatives_func,
            args=args,
            term=self.term,
            solver=self.diffrax_solver,
            max_steps=self.solver_args["max_steps"],
        )

    def _gillespie_wrapper(
        self,
        y0: jnp.ndarray,
        dt: float,
        derivatives_func: Callable[..., jnp.ndarray],
        args: Tuple[Any, ...],
    ) -> jnp.ndarray:
        return self._run_gillespie(y0, dt, args)

    def _run_gillespie(
        self,
        y0: jnp.ndarray,
        dt: float,
        args: Tuple[Any, ...],
    ) -> jnp.ndarray:
        """
        Simple SSA (Gillespie) over [0, dt], holding V constant.
        Requires self.n_channels on the channel.
        """
        N = getattr(self, "n_channels", None)
        if N is None:
            raise RuntimeError("You must set `.n_channels` on the channel before using `gillespie`.")

        counts = np.round(np.array(y0) * N).astype(int)
        V = args[0]

        # Precompute gate rates at this V
        αm, βm = self.m_gate(V)
        try:
            αh, βh = self.h_gate(V)
        except AttributeError:
            αh = βh = 0.0

        t = 0.0
        while True:
            if t >= dt:
                break

            # Build rate list depending on #states
            rates = []
            if counts.size == 8:
                # Na8States: [C3,C2,C1,O,I3,I2,I1,I]
                C3,C2,C1,O,I3,I2,I1,I = counts
                rates = [
                    3*αm*C3, 2*αm*C2, 1*αm*C1, 3*βm*O,
                    βh*C3,   βh*C2,   βh*C1,   βh*O,
                    αh*I3,   αh*I2,   αh*I1,   αh*I,
                ]
            else:
                # K5States: [C4,C3,C2,C1,O]
                C4,C3,C2,C1,O = counts
                rates = [
                    4*αm*C4, 3*αm*C3, 2*αm*C2, 1*αm*C1,
                    4*βm*O,
                ]

            Rtot = sum(rates)
            if Rtot <= 0:
                break

            u1, u2 = np.random.rand(), np.random.rand()
            tau = -np.log(u1)/Rtot
            if t + tau > dt:
                break

            idx = int(np.searchsorted(np.cumsum(rates), u2*Rtot))
            # apply state jump
            if counts.size == 8:
                # Na8States transitions
                if idx==0:  counts[0]-=1; counts[1]+=1
                elif idx==1:counts[1]-=1; counts[2]+=1
                elif idx==2:counts[2]-=1; counts[3]+=1
                elif idx==3:counts[3]-=1; counts[2]+=1
                elif idx==4:counts[0]-=1; counts[4]+=1
                elif idx==5:counts[1]-=1; counts[5]+=1
                elif idx==6:counts[2]-=1; counts[6]+=1
                elif idx==7:counts[3]-=1; counts[7]+=1
                elif idx==8:counts[4]-=1; counts[0]+=1
                elif idx==9:counts[5]-=1; counts[1]+=1
                elif idx==10:counts[6]-=1;counts[2]+=1
                elif idx==11:counts[7]-=1;counts[3]+=1
            else:
                # K5States transitions
                if   idx==0: counts[0]-=1; counts[1]+=1
                elif idx==1: counts[1]-=1; counts[2]+=1
                elif idx==2: counts[2]-=1; counts[3]+=1
                elif idx==3: counts[3]-=1; counts[4]+=1
                elif idx==4: counts[4]-=1; counts[3]+=1

            t += tau

        return jnp.array(counts/float(N))


### — Deterministic / SDE solvers —

def explicit_euler(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    *args: Any
) -> jnp.ndarray:
    return y0 + derivatives_func(None, y0, *args) * dt


def newton(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    *args: Any,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 10
) -> jnp.ndarray:
    def _f(y, y_prev): return y - y_prev - dt*derivatives_func(None, y, *args)
    def body(carry):
        y, y0_, i, _ = carry
        F = _f(y,y0_)
        J = jax.jacobian(_f)(y,y0_).reshape((y.size,y.size))
        δ = solve(J, -F.flatten()).reshape(y.shape)
        y_new = y + δ
        conv = jnp.linalg.norm(δ) < (atol + rtol*jnp.linalg.norm(y_new))
        return (y_new, y0_, i+1, conv)
    def cond(carry):
        _,_,i,conv = carry
        return (i<max_steps)&~conv
    y_final,_,_,_ = eqx.internal.while_loop(cond, body, (y0,y0,0,False),
                                           max_steps=max_steps, kind="checkpointed")
    return y_final


def rk45(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    *args: Any
) -> jnp.ndarray:
    def f(y): return derivatives_func(None,y,*args)
    a2,a3   = 1/4, [3/32,9/32]
    a4      = [1932/2197,-7200/2197,7296/2197]
    a5      = [439/216,-8,3680/513,-845/4104]
    a6      = [-8/27,2,-3544/2565,1859/4104,-11/40]
    b5      = [16/135,0,6656/12825,28561/56430,-9/50,2/55]
    k1      = f(y0)
    k2      = f(y0 + a2*dt*k1)
    k3      = f(y0 + dt*(a3[0]*k1+a3[1]*k2))
    k4      = f(y0 + dt*(a4[0]*k1+a4[1]*k2+a4[2]*k3))
    k5      = f(y0 + dt*(a5[0]*k1+a5[1]*k2+a5[2]*k3+a5[3]*k4))
    k6      = f(y0 + dt*(a6[0]*k1+a6[1]*k2+a6[2]*k3+a6[3]*k4+a6[4]*k5))
    return y0 + dt*(b5[0]*k1 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6)


def diffrax_implicit(
    y0: jnp.ndarray,
    dt: float,
    derivatives_func: Callable[..., jnp.ndarray],
    args: Tuple[Any, ...],
    term: ODETerm,
    solver: ImplicitEuler,
    max_steps: int,
) -> jnp.ndarray:
    sol = diffeqsolve(term, solver, args=args, t0=0.0, t1=dt, dt0=dt,
                      y0=y0, max_steps=max_steps)
    return jnp.squeeze(sol.ys, axis=0)
