# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from typing import Literal, Union

from scipy.sparse import issparse, csr_matrix, vstack, hstack, eye, spmatrix
from scipy.optimize import linprog

from ortools.linear_solver import pywraplp

from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.estimation.algorithm.base import BaseAlgorithm
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra

import logging
std_logger = logging.getLogger(__name__)
std_logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

LinprogMethod = Literal["highs", "highs-ds", "highs-ipm"]
_ALLOWED_LINPROG_METHODS = {"highs", "highs-ds", "highs-ipm"}
ArrayLike = Union[NDArray[np.float64], spmatrix]

class LPAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance: float, maximum_iterations: int, logger: logging.Logger = std_logger) -> None:
        r"""
        The algorithm solves a (weighted) 'Least Absolute Value (LAV)' optimization problem to estimate the system
        state vector from possibly bad or noisy measurements.

        Parameters:
            tolerance:
                Convergence threshold for the state update :math:`\lVert \Delta E \rVert_{\infty}`. The iterative
                process stops once the maximum absolute update is below this value.
            maximum_iterations:
                Maximum number of iterations allowed before the algorithm is considered not converged.
            logger:
                Logger instance used for diagnostic and error messages.
        """
        # Initialize base algorithm
        super(LPAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # store results for diagnostics
        self.r = None  # residual z-h(x)
        self.H = None  # Jacobian matrix
        self.hx = None  # calculated measurements h(x)
        self.obj_func = None  # objective function J(x)
        self.af: pd.DataFrame | None = None  # calculation results from allocation factor

    def estimate(
            self,
            eppci: ExtendedPPCI,
            debug_mode=False,
            linprog_method: LinprogMethod = "highs",
            wlav: bool = False,
            with_ortools: bool = True,
            with_af_constraints: bool = True,
            **kwargs
    ) -> ExtendedPPCI | bool:
        r"""
        Perform power system state estimation using the (weighted) least absolute value (LAV/WLAV) formulation.

        The method solves an iterative linear programming problem based on the linearized measurement model

        .. math::
            r = z - h(x), \quad r_{\text{new}} \approx r - H \,\Delta E,

        :math:`z` is the measurement vector, :math:`h(x)` is the nonlinear measurement function and :math:`r` is the
        residual vector. In the script the current state vector :math:`x = E` and
        :math:`H` is the Jacobian-Matrix.

        For the unweighted LAV estimator, the objective is to minimize the 1-norm of the residual vector:

        .. math::
            J_{\mathrm{LAV}} = \sum_{i=1}^{m} |r_i|.

        For the weighted LAV (WLAV) estimator, each residual is normalized by the standard deviation :math:`\sigma_i` of
        the corresponding measurement:

        .. math::
            J_{\mathrm{WLAV}} = \sum_{i=1}^{m} \left|\frac{r_i}{\sigma_i}\right|
            = \sum_{i=1}^{m} \frac{1}{\sigma_i}|r_i|.

        Thus, the weights used by the WLAV estimator are

        .. math::
            w_i = \frac{1}{\sigma_i},

        whereas the unweighted LAV estimator uses :math:`w_i = 1`. With the goal to minimize

        .. math::
            \min \sum_i w_i \, \lvert r_i \rvert.

        In each iteration, the nonlinear measurement model is linearized around the current state and a linear
        programming problem is solved via `SciPy <https://docs.scipy.org/doc/scipy/>`_ or
        `OR-Tools solver <https://github.com/google/or-tools>`_ to determine the state update :math:`\Delta E` in
        ``eppci``. The state vector is subsequently updated according to

        .. math::
            E \leftarrow E + \Delta E

        with the state vector :math:`E = [\theta_{\mathrm{non-slack}}, V_{\mathrm{all-buses}}]^\top`.

        Parameters:
            eppci:
                Central data container (ExtendedPPCI) containing the network model, measurements, current state vector
                ``E``, measurement vector ``z``, and (optionally) covariance information ``r_cov`` for WLAV.
            debug_mode:
                If ``True``, additional diagnostic information is logged, including the current state update norm and
                the current LAV objective value.
            linprog_method:
                Method name passed to :func:`scipy.optimize.linprog` (e.g. ``"highs"``).
            wlav:
                If ``True``, perform weighted LAV, where the weights are computed as ``1 / sigma`` with
                ``sigma = max(r_cov, 1e-5)``. If ``False``, all measurements are weighted equally.
            with_ortools:
                If ``True``, use the `OR-Tools solver <https://github.com/google/or-tools>`_
            with_af_constraints:
                Constraints for allocation factors between 0 and 1 for (W)LAV algorithm.

        Keyword Arguments:
            **kwargs: Currently unused. Present for API compatibility and possible future extensions.

        Returns:
            The updated data container with the estimated state variables if the optimization is successful.
            Additionally, on success the following attributes are populated for diagnostics:

            * ``self.r``: final residual vector :math:`z - h(E)`
            * ``self.H``: final Jacobian matrix at the estimated state
            * ``self.hx``: final calculated measurements :math:`h(E)`
            * ``self.obj_func``: final LAV objective value
            * ``self.iterations``: number of iterations performed

            Returns ``False`` if the optimization fails or an exception occurs.
        """
        # initialize eppci and check the observability
        self.initialize(eppci)

        # matrix calculation object for the state estimation parameter
        sem = BaseAlgebra(eppci)
        current_error = 100.
        cur_it = 0  # is separate and later set to self.iterations. Reason…
        E = eppci.E

        af_lp: bool = (eppci.algorithm == "af-lp")

        while current_error > self.tolerance and cur_it < self.max_iterations:
            try:
                # residual r=z-h(x)
                r = LPAlgorithm._to_dense_auto(sem.create_rx(E))
                # create Jacobian matrix convert to csr -> zeros not save -> better for lager grids -> less RAM
                H_raw = sem.create_hx_jacobian(E)  # create jacobian matrix from data set
                H = H_raw.tocsr() if issparse(H_raw) else csr_matrix(H_raw)  # H_raw has to be a SciPy-CSR-Matrix
                # m number of measurements -> z element R^{m}, n number of state variable
                m, n = H.shape

                # set bounds depend on allocation factor for solving the linear programming problem
                if af_lp:
                    # u_i >= 0, dE free, 0 <= alpha_i <= 1
                    if with_af_constraints:
                        bounds = LPAlgorithm._create_af_bounds(n, m, len(self.eppci["clusters"]), E)
                    else:
                        bounds = [(None, None)] * n + [(0, None)] * m  # for debugging and to compare results ToDo: remove after compare and debugging
                else:
                    # u_i >= 0, dE free
                    bounds = [(None, None)] * n + [(0, None)] * m

                if wlav:
                    # check that sigma is not near to zero prevent 1/0
                    sigma = np.maximum(LPAlgorithm._to_dense_auto(eppci.r_cov), 1e-5)
                    weights = 1.0 / sigma
                else:
                    weights = np.ones(m)

                if with_ortools:
                    d_E, obj_value = self._solve_lav_ortools(H, r, weights, bounds)
                else:
                    d_E, obj_value = self._solve_lav_scipy(H, r, weights, linprog_method, bounds)

                current_error = float(np.max(np.abs(d_E)))

                # Optional step limiting:
                # This factor was derived from a project, where a flat start in large-scale grids caused
                # excessively large state updates (dE). Limiting the step to 0.35 helps prevent those large jumps during
                # the first iterations.
                if current_error > 0.35:  # project heuristic to limit excessive state updates
                    d_E = d_E * 0.35 / current_error

                # Update state vector
                E += d_E
                eppci.update_E(E)

                if debug_mode:
                    self.obj_func = obj_value
                    self.logger.debug(f"Current delta_x: {current_error:.7f}")
                    self.logger.debug(f"Current LAV objective value: {obj_value:.7f}")

                cur_it += 1

            except Exception as err:
                self.logger.error(f"A problem appeared while running LAV estimation: {err}")
                return False

        self.check_result(current_error, cur_it)  # set self.successful to true if se works
        self.iterations = cur_it

        if debug_mode:
            print(f"number of required iterations: {cur_it} of {self.max_iterations}")
            print(f"current_error = {current_error}, threshold = {self.tolerance}")

        if self.successful:
            self.H = np.asarray(sem.create_hx_jacobian(E))  # ToDo: E or eppci.E
            self.hx = sem.create_hx(E)  # ToDo: Which E or eppci.E should be used
            self.r = np.asarray(sem.create_rx(E)).reshape(-1)
        """
        Attention: Bad side-effects
        this works:
            self.H = np.asarray(sem.create_hx_jacobian(eppci.E))
            self.hx = sem.create_hx(E)
            self.r = np.asarray(sem.create_rx(E)).reshape(-1)
        this does not work:
            self.hx =self.hx = sem.create_hx(E)
            self.r = np.asarray(sem.create_rx(E)).reshape(-1)
            self.H = np.asarray(sem.create_hx_jacobian(eppci.E))
        """

        # split voltage and allocation factor variables
        # clusters are set in ppc_conversion.py/_add_rated_power_information_af_wls() only if algorithm == "af-wls"
        if "clusters" in self.eppci:
            num_clusters = len(self.eppci["clusters"])
            E1 = E[:-num_clusters]
            E2 = E[-num_clusters:]
            self.af = pd.DataFrame([E2], columns=eppci["clusters"])
            eppci.update_E(E1)
            eppci.clusters = E2

        return eppci

    @staticmethod
    def _create_af_bounds(
            n: int,
            m: int,
            num_clusters: int,
            E: NDArray[np.float64]
    ) -> list[tuple[float, None] | tuple[float, float] | tuple[None, None]]:
        r"""
        Create variable bounds for the allocation factors LAV/WLAV linear programming problem.

        The LP variable vector is:

        .. math::
            y = [\Delta E_1, \ldots, \Delta E_n, u_1, \ldots, u_m]^\top

        with:

        .. math::
            E = [\theta_{\mathrm{non-slack}}, V_{\mathrm{all-buses}}, \alpha_{i}]^\top

        For standard state variables, the state update ``dE`` = :math:`\Delta E` is unconstrained. If allocation factors are present, they
        are assumed to be the last :math:`i` ``num_clusters`` entries of ``E`` = :math:`E` and are constrained such that:

        .. math::
            0 \le \alpha_{\mathrm{old}} + \Delta\alpha \le 1

        which leads to:

            .. math::
                -\alpha_{\mathrm{old}} \le \Delta\alpha \le 1 - \alpha_{\mathrm{old}}

        The auxiliary variables :math:`u` are constrained to be nonnegative.

        Parameters:
            n:
                Number of state variables.
            m:
                Number of measurements.
            num_clusters:
                Number of different clusters in power grid.
            E:
                current state vector with :math:`E = [\theta_{\mathrm{non-slack}}, V_{\mathrm{all-buses}}, \alpha_{i}]`
                the values :math:`\alpha_{i}` are existing if the allocation factors are used.
        Returns:
            Bounds list for ``scipy.optimize.linprog``.
        """
        af_bounds = []
        eps = 1e-6  # to prevent small negative number something like -3.45e-27
        alpha_old = E[-num_clusters:]
        for k in range(num_clusters):
            lower_bound = eps - alpha_old[k]
            upper_bound = 1. - alpha_old[k]
            af_bounds.append((float(lower_bound), float(upper_bound)))
        bounds = [(None, None)] * (n - num_clusters) + af_bounds + [(0, None)] * m
        return bounds

    @staticmethod
    def _to_dense_auto(x: ArrayLike) -> np.ndarray:
        """
        Convert input to a dense NumPy array and automatically adjust its shape.

        Sparse matrices are converted using :func:`toarray()`. Other array-like inputs are converted using
        :func:`numpy.asarray`. Vector-like arrays with shape ``(n,)``, ``(n, 1)``, or ``(1, n)`` are returned as
        one-dimensional arrays. Matrix-like arrays are returned unchanged.

        Parameters:
            x: Input data, such as a NumPy array, list, or SciPy sparse matrix.

        Returns:
            Dense NumPy array. Vector-like inputs are flattened to shape ``(n,)``.
        """
        arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        arr = np.asarray(arr, dtype=np.float64)

        if arr.ndim == 0:
            return arr.reshape(1)

        if arr.ndim == 1:
            return arr

        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return arr.reshape(-1)

        return arr

    @staticmethod
    def _report_range(name: str, values: ArrayLike) -> None:
        """
        Log basic numerical information about an array.
        """
        # Convert sparse and dense inputs to a one-dimensional NumPy array
        if hasattr(values, "data") and issparse(values):
            values = values.data

        values = np.asarray(values, dtype=np.float64).reshape(-1)

        if values.size == 0:
            std_logger.debug("%s: empty array", name)
            return

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            std_logger.debug("%s: shape=%s, contains no finite values", name, values.shape)
            return

        std_logger.debug(
            "%s: shape=%s, min=%g, max=%g, max_abs=%g",
            name,
            values.shape,
            np.min(finite_values),
            np.max(finite_values),
            np.max(np.abs(finite_values)),
        )

    @staticmethod
    def _solve_lav_scipy(
            H: csr_matrix,
            r: NDArray[np.float64],
            weights: NDArray[np.float64],
            linprog_method: LinprogMethod,
            bounds: list[tuple[float, None] | tuple[float, float] | tuple[None, None]]
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Solve the LAV/WLAV linear programming problem using :func:`scipy.optimize.linprog`.

        The optimization problem is formulated as:

        .. math::
            \min \sum_i w_i \, u_i

        subject to:

        .. math::
            -u \le r - H \Delta E \le u

        where:

            - :math:`\Delta E` = ``dE`` are the state updates
            - :math:`u_i` are auxiliary nonnegative variables representing the absolute residuals
            - :math:`r` = ``r`` is the current residual vector
            - :math:`H` = ``H`` is the Jacobian matrix

        The LP variable vector is defined as:

        .. math::
            y = [\Delta E_1, \ldots, \Delta E_n, u_1, \ldots, u_m]^\top

        The inequality constraints are assembled into the standard LP form:

        .. math::
            A_{\mathrm{ub}} \, y \le b_{\mathrm{ub}}

        with:

        .. math::
            A_{\mathrm{ub}} = \begin{bmatrix} -H & -I \\ H & -I \end{bmatrix}

        and:

        .. math::
            b_{\mathrm{ub}} = \begin{bmatrix} -r \\ r \end{bmatrix}

        which corresponds to:

        .. math::
            -H \Delta E - u \le -r

        .. math::
             H \Delta E - u \le r

        Sparse matrices are used throughout the formulation to improve performance and memory efficiency for large-scale
        power systems. For the allocation factors :math:`E` is defined as:

        .. math::
            E = [\theta_{\mathrm{non-slack}}, V_{\mathrm{all-buses}}, \alpha_{i}]^\top

        with :math:`i` number of clusters.

        Parameters:
            H: Sparse Jacobian matrix with shape ``(m, n)``.
            r: Residual vector ``z - h(x)`` with shape ``(m,)``.
            weights: Weight vector for WLAV. For standard LAV, this is typically ``np.ones(m)``.
            linprog_method: Method passed to ``scipy.optimize.linprog`` (e.g. ``"highs"``).
            bounds: Bounds for solve minimization. :math:`0 \le u_{i}`, dE free, 0 <= alpha_i <= 1

        Raises:
            numpy.linalg.LinAlgError: If the LP optimization fails or no feasible solution is found.

        Returns:
            a tuple with

                * :math:`\Delta E`: State update vector with shape ``(n,)``
                * ``objective_value``: Final value of the LP objective function.
        """
        # m number of measurements len(eppci.z) -> z element R^{m}, n number of state variable len(eppci.E)
        m, n = H.shape

        # Validate the shape of the residual vector
        if r.shape != (m,):
            raise ValueError(f"Invalid shape of r: {r.shape}, expected {(m,)}")

        # Validate the shape of the weight vector
        if weights.shape != (m,):
            raise ValueError(f"Invalid shape of weights: {weights.shape}, expected {(m,)}")

        # The bounds must cover all state-update and auxiliary variables
        if len(bounds) != n + m:
            raise ValueError(f"Invalid number of bounds: {len(bounds)}, expected {n + m}")

        # Check the residual vector for NaN or infinite values
        if not np.isfinite(r).all():
            raise ValueError("r contains NaN or Inf.")

        # Check the non-zero entries of the sparse Jacobian matrix
        if not np.isfinite(H.data).all():
            raise ValueError("H contains NaN or Inf.")

        # Check the weights for NaN or infinite values
        if not np.isfinite(weights).all():
            raise ValueError("weights contains NaN or Inf.")

        # Validate every lower and upper bound
        for index, (lower, upper) in enumerate(bounds):
            # Lower bounds must be finite if they are specified
            if lower is not None and not np.isfinite(lower):
                raise ValueError(f"Invalid lower bound at index {index}: {lower}")

            # Upper bounds must be finite if they are specified
            if upper is not None and not np.isfinite(upper):
                raise ValueError(f"Invalid upper bound at index {index}: {upper}")

            # The lower bound must not exceed the upper bound
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"Infeasible bound at index {index}: ({lower}, {upper})")



        # We solve (Abur - Power system state estimation: theory and implementation, 2004):
        # min sum(abs(r_i)) -> abs non-linear -> -u_i <= r <= u_i and r_i = z_i - h_i(x)
        # r_i -> risidual of iteration i (differenc between measurement values and model function)
        # h_i(x) is nonlinear, we like to get the best state x with an iterative linear approach
        # r_i^{new} = z_i - h_i(x + dx) = z_i - (h_i(x) + H dx) = r_i - H_i dx
        # -> -u_i <= r_i - H_i dx <= u_i  # Attention pandapower E = x

        # min sum(u_i)
        # subject to:
        # -u_i <= r_i - H_i * dE <= u_i

        # linprog (scipy) -> -u_i <= r_i - H_i * dE <= u_i -> 2 constraints among themselves
        # Variable vector:
        # y = [dE_1, ..., dE_n, u_1, ..., u_m]
        # In the unweighted case, c_i(dE_i) = 0 and c_i(E) = 1

        c = np.r_[np.zeros(n), weights]

        # Constraint:
        # r - H dE <= u
        # -H dE - u <= -r
        I = eye(m, format="csr")  # Sparse identity instead of dense np.eye(m)
        A1 = hstack([-H, -I], format="csr")
        b1 = -r

        # Constraint:
        # -(r - H dE) <= u
        # H dE - u <= r
        A2 = hstack([H, -I], format="csr")
        b2 = r

        # ub stands for upper bound -> A_ub * y <= b_ub
        A_ub = vstack([A1, A2], format="csr")
        b_ub = np.r_[b1, b2]

        # Report numerical ranges before calling HiGHS
        LPAlgorithm._report_range("H.data", H.data)
        LPAlgorithm._report_range("r", r)
        LPAlgorithm._report_range("weights", weights)
        LPAlgorithm._report_range("c", c)
        LPAlgorithm._report_range("b_ub", b_ub)

        std_logger.debug(
            "LP dimensions: H=%s, A_ub=%s, b_ub=%s, c=%s, bounds=%d",
            H.shape,
            A_ub.shape,
            b_ub.shape,
            c.shape,
            len(bounds),
        )

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method=linprog_method
        )

        if not result.success:
            raise np.linalg.LinAlgError(
                f"SciPy LAV optimization failed:"
                f"status={result.status},"
                f"message={result.message},"
                f"success={result.success}"
            )
        if result.x is None:
            raise np.linalg.LinAlgError("SciPy LAV optimization returned no solution vector.")
        if result.fun is None:
            raise np.linalg.LinAlgError("SciPy LAV optimization returned no objective value.")

        # Extract state update
        d_E: NDArray[np.float64] = np.asarray(result.x[:n], dtype=np.float64)
        objective_value: float = float(result.fun)

        return d_E, objective_value

    @staticmethod
    def _solve_lav_ortools(
            H: csr_matrix,
            r: NDArray[np.float64],
            weights: NDArray[np.float64],
            bounds: list[tuple[float, None] | tuple[float, float] | tuple[None, None]]
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Solve the (weighted) Least Absolute Value (LAV/WLAV) state estimation subproblem using
        `OR-Tools SCIP <https://github.com/google/or-tools>`_.

        The nonlinear measurement model is linearized around the current state:

        .. math::
            r^{new} \approx r - H \Delta E

        where:

            - :math:`r` is the current residual vector
            - :math:`H` is the Jacobian matrix
            - :math:`\Delta E` is the state update vector

        The optimization problem minimizes the weighted absolute residuals:

        .. math::
            \min \sum_i w_i |r_i - H_i \Delta E|

        Since the absolute value operator is nonlinear, auxiliary variables :math:`u_i` are introduced such that:

        .. math::
            -u_i \le r_i - H_i \Delta E \le u_i

        The LP variable vector is defined as:

        .. math::
            y = [\Delta E_1, \ldots, \Delta E_n, u_1, \ldots, u_m]^\top

        The optimization problem can then be written in standard LP form:

        .. math::
            \min \sum_i w_i u_i

        subject to:

        .. math::
            A_{ub} \, y \le b_{ub}

        with:

        .. math::
            A_{ub} = \begin{bmatrix} H & -I \\ -H & -I \end{bmatrix}

        and:

        .. math::
            b_{ub} = \begin{bmatrix} r \\ -r \end{bmatrix}

        which corresponds to the pair of inequalities:

        .. math::
             H \Delta E - u \le r

        .. math::
            -H \Delta E - u \le -r

        Notes:
            * ``dE`` variables are free variables and may take positive or negative values
            * ``u`` variables are nonnegative and represent the absolute residual magnitudes
            * The Jacobian matrix is processed row-wise in sparse CSR format
              for efficiency on large-scale sparse systems


        Parameters:
            H:
                Sparse Jacobian matrix with shape ``(m, n)``. ``m`` = number of measurements and ``n`` = number of
                state variables

            r:
                Current residual vector: :math:`r = z - h(E)` with shape ``(m,)``.

            weights:
                Weight vector used for WLAV. Typical choices ``weights = np.ones(m)`` (Standard LAV) and
                ``weights = 1 / sigma`` (Weighted LAV)

            bounds:
                Bounds for solve minimization. :math:`0 \le u_{i}`, dE free, 0 <= alpha_i <= 1


        Raises:
            numpy.linalg.LinAlgError:
                If OR-Tools SCIP is unavailable or if the optimization fails.

        Returns:
            tuple containing:

            * **d_E**
              State update vector with shape ``(n,)``.

            * **objective_value**
              Final LP objective value:

              .. math::
                  \sum_i w_i u_i
        """
        # m number of measurements -> z element R^{m}, n number of state variable
        m, n = H.shape

        # Small threshold to ignore tiny numerical coefficients when building sparse linear expressions
        error_margin = 1e-10

        # Create OR-Tools SCIP solver instance
        #
        # SCIP is generally robust for sparse LP problems arising in power system state estimation
        solver = pywraplp.Solver.CreateSolver("SCIP")

        if solver is None:
            raise np.linalg.LinAlgError("OR-Tools SCIP solver is not available.")

        infinity = solver.infinity()

        # ------------------------------------------------------------------
        # Create optimization variables
        # ------------------------------------------------------------------

        # State update variables:
        #
        # bounds contains bounds for the full LP vector:
        # y = [dE_1, ..., dE_n, u_1, ..., u_m]
        #
        # Therefore, the first n entries belong to dE.
        dE = []
        for j in range(n):
            lower, upper = bounds[j]

            lb = -infinity if lower is None else float(lower)
            ub = infinity if upper is None else float(upper)

            dE.append(solver.NumVar(lb, ub, f"dE_{j}"))

        # Auxiliary residual magnitude variables:
        # u_i >= 0
        # These variables represent:
        # |r_i - H_i dE|
        # and are minimized in the objective function
        u = []
        for i in range(m):
            lower, upper = bounds[n + i]

            lb = 0.0 if lower is None else float(lower)
            ub = infinity if upper is None else float(upper)

            u.append(solver.NumVar(lb, ub, f"u_{i}"))

        # ------------------------------------------------------------------
        # Add inequality constraints
        # ------------------------------------------------------------------

        # Iterate row-wise through the sparse Jacobian matrix. Each row corresponds to one measurement equation
        for i in range(m):
            # CSR row access:
            #
            # indptr stores row boundaries
            # Row i contains entries:
            # indices[start:end]
            # data[start:end]
            start, end = H.indptr[i], H.indptr[i + 1]

            cols = H.indices[start:end]
            vals = H.data[start:end]

            # Build sparse linear expression:
            #
            # H_i dE = Σ_j H_ij * dE_j
            #
            # Only nonzero Jacobian entries are processed
            h_expr = solver.Sum(vals[k] * dE[cols[k]] for k in range(len(vals)) if abs(vals[k]) > error_margin)

            # Constraint:
            # H_i dE - u_i <= r_i
            solver.Add(h_expr - u[i] <= float(r[i]))

            # Constraint:
            # -H_i dE - u_i <= -r_i
            solver.Add(-h_expr - u[i] <= float(-r[i]))

        # ------------------------------------------------------------------
        # Objective function
        # ------------------------------------------------------------------

        # Minimize weighted sum of auxiliary variables:
        # min Σ_i weights_i * u_i

        # Since u_i >= |r_i - H_i dE|,
        # this minimizes the weighted L1 norm
        objective = solver.Sum(float(weights[i]) * u[i] for i in range(m))

        solver.Minimize(objective)

        # ------------------------------------------------------------------
        # Solve optimization problem
        # ------------------------------------------------------------------
        status = solver.Solve()

        # Accept:
        #
        # OPTIMAL  -> globally optimal LP solution found
        # FEASIBLE -> feasible solution found
        if status not in (
                pywraplp.Solver.OPTIMAL,
                pywraplp.Solver.FEASIBLE
        ):
            raise np.linalg.LinAlgError(
                "OR-Tools LAV optimization failed."
            )

        # ------------------------------------------------------------------
        # Extract optimized state update vector
        # ------------------------------------------------------------------
        d_E: NDArray[np.float64] = np.asarray([var.solution_value() for var in dE], dtype=np.float64)
        # Final objective value: Σ_i weights_i * u_i
        objective_value = solver.Objective().Value()
        return d_E, float(objective_value)
