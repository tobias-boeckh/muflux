from copy import deepcopy

import iminuit
import numpy as np
from iminuit import Minuit
from scipy.special import loggamma
from uncertainties import unumpy as unp


class Fit:
    def __init__(
        self,
        meas: np.ndarray,
        initial_guess: dict[str, np.ndarray],
        mig: np.ndarray | None = None,
        mig_unc: np.ndarray | None = None,
        eff: np.ndarray | None = None,
        eff_unc: np.ndarray | None = None,
        fixed_params: dict[str, float] | None = None,
        regularization: float = 0.0,
        reg_type: str = "log_curvature",
    ) -> None:
        self._meas = meas
        self._initial_guess = initial_guess
        self._regularization = regularization
        self._reg_type = reg_type
        self._mig = mig
        self._mig_unc = mig_unc
        self._eff = eff
        self._eff_unc = eff_unc

        self._size = len(meas)
        self._has_eff_unc = self._eff_unc is not None
        self._has_mig_unc = self._mig_unc is not None
        self._names = self.__var_helper("mu")

        if self._eff is None:
            self._eff = np.ones(self._size)
        if self._has_eff_unc:
            self._names += [f"theta_eff_{i}" for i in range(self._size)]
        else:
            self._eff_unc = np.zeros_like(self._eff)
        if self._mig is None:
            self._mig = np.eye(self._size)
        if self._has_mig_unc:
            self._names += [f"theta_mig_{i}" for i in range(self._size * self._size)]
        else:
            self._mig_unc = np.zeros_like(self._mig)
        self._names = np.hstack(self._names)

        # setup parameter indices
        self._mu_indices = self.__get_index("mu")
        self._theta_eff_indices = self.__get_index("theta_eff")
        self._theta_mig_indices = self.__get_index("theta_mig")

        self._setup_fit(fixed_params)
        self._global_fit()

    def _setup_fit(self, fixed_params: dict[str, float] | None = None) -> None:
        """Setup likelihood, minuit, error definition and parameter limits"""

        # get values from initial guess dict in correct order
        initial_guess = [*self._initial_guess["mu"]]
        if self._has_eff_unc:
            initial_guess += [
                self._initial_guess[f"theta_eff_{i}"] for i in range(self._size)
            ]
        if self._has_mig_unc:
            initial_guess += [
                self._initial_guess[f"theta_mig_{i}"]
                for i in range(self._size * self._size)
            ]

        self.__m = Minuit(
            self.__likelihood_wrapper, np.array(initial_guess), name=self._names
        )
        self.__m.errordef = Minuit.LEAST_SQUARES  # 1.0
        self.__m.limits[self.__var_helper("mu")] = (0, None)

        # fix theta_mig parameters to zero where the nominal migration matrix is zero
        if self._has_mig_unc:
            n_fixed = 0
            for i, val in enumerate(self._mig.flatten()):
                if val < 1e-3:
                    # Fix small values, which don't contribute but slow down fit
                    self.__m.values[f"theta_mig_{i}"] = 0.0
                    self.__m.fixed[f"theta_mig_{i}"] = True
                    n_fixed += 1

        # fix values to optimal fit value for stat. uncertainty fit
        if fixed_params is not None:
            for param, value in fixed_params.items():
                self.__m.values[param] = value
                self.__m.fixed[param] = True

    def __var_helper(self, var: str) -> list[str]:
        """Create list of variables of all bins"""
        return [f"{var}_{i}" for i in range(self._size)]

    def __get_index(self, var: str) -> np.ndarray:
        return np.array([name.startswith(var) for name in self._names])

    def __likelihood_wrapper(self, pars: np.ndarray) -> float:
        """Get fit parameters from flat array and call likelihood function"""
        pars_dict = {"mu": pars[self._mu_indices]}
        if self._has_eff_unc:
            pars_dict["theta_eff"] = pars[self._theta_eff_indices]
        if self._has_mig_unc:
            pars_dict["theta_mig"] = pars[self._theta_mig_indices]
        return self.__likelihood(**pars_dict)

    @staticmethod
    def __log_poisson(k: np.ndarray, nu: np.ndarray) -> np.ndarray:
        nu = np.maximum(nu, 1e-8)
        return k * np.log(nu) - nu - loggamma(k + 1)

    def __likelihood(
        self,
        mu: np.ndarray,
        theta_eff: np.ndarray | None = None,
        theta_mig: np.ndarray | None = None,
    ) -> float:
        if self._has_eff_unc:
            _eff = self._eff * (
                np.ones_like(self._eff)
                + np.sum(theta_eff.reshape(-1, self._size) * self._eff_unc, axis=0)
            )
        else:
            _eff = self._eff

        if self._has_mig_unc:
            _mig = self._mig + np.sum(
                theta_mig.reshape(-1, self._size, self._size) * self._mig_unc,
                axis=0,
            )
        else:
            _mig = self._mig

        _n0 = _mig @ (_eff * mu)
        nll = -2 * np.sum(self.__log_poisson(self._meas, _n0))
        if self._has_eff_unc:
            nll += np.sum(theta_eff**2)
        if self._has_mig_unc:
            nll += np.sum(theta_mig**2)
        if self._regularization > 0.0 and len(mu) > 2:
            if self._reg_type == "log_curvature":
                log_mu = np.log(np.maximum(mu, 1e-10))
                nll += self._regularization * np.sum(
                    (log_mu[:-2] - 2 * log_mu[1:-1] + log_mu[2:]) ** 2
                )
            elif self._reg_type == "curvature":
                nll += self._regularization * np.sum(
                    (mu[:-2] - 2 * mu[1:-1] + mu[2:]) ** 2
                )
        return nll

    def __fit(self) -> None:
        """Run iminuit fit"""
        self.__m.simplex()
        self.__m.migrad()
        self.__m.minos()
        self.__m.hesse()

    def _global_fit(self) -> None:
        """Run fit for all parameters"""
        self.__m.fixed[self.__var_helper("mu")] = False
        self.__fit()
        self.__mglobal = deepcopy(self.__m)

    def _background_fit(self) -> None:
        """Run fit with signal parameters fixed to 0"""
        self.__m.values[self.__var_helper("mu")] = 0
        self.__m.fixed[self.__var_helper("mu")] = True
        self.__fit()
        self.__mbgr = deepcopy(self.__m)

    def stat_uncertainty_fit(self, fit_param_dict):
        self.__m.fixed[self.__var_helper("mu")] = False
        for param, value in fit_param_dict.items():
            self.__m.values[param] = value
            self.__m.fixed[param] = True
        self.__fit()
        return deepcopy(self.__m)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def mglobal(self) -> iminuit.minuit.Minuit:
        if self.__mglobal is None:
            self._global_fit()
        return self.__mglobal

    @property
    def m(self) -> iminuit.minuit.Minuit:
        return self.mglobal

    @property
    def mbgr(self) -> iminuit.minuit.Minuit:
        if self.__mbgr is None:
            self._background_fit()
        return self.__mbgr

    @property
    def significance(self) -> float:
        return np.sqrt(self.mbgr.fval - self.mglobal.fval)

    @property
    def values(self) -> iminuit.util.ValueView:
        return self.mglobal.values

    @property
    def errors(self) -> iminuit.util.ErrorView:
        return self.mglobal.errors

    @property
    def covariance(self) -> iminuit.util.Matrix:
        return self.mglobal.covariance

    @property
    def merrors(self) -> tuple[np.ndarray, np.ndarray]:
        merrors = np.array(list(self.m.merrors.values()))
        lower = np.array([np.abs(i.lower) for i in merrors])
        upper = np.array([np.abs(i.upper) for i in merrors])
        return lower, upper

    @property
    def muarray(self) -> np.ndarray:
        minos_keys = list(self.m.merrors.keys())
        max_merrors = np.max(np.vstack(self.merrors), axis=0)
        uncertainties = np.zeros(len(self.values))
        for i, name in enumerate(minos_keys):
            uncertainties[self.names.index(name)] = max_merrors[i]
        return unp.uarray(np.array(self.values), uncertainties)

    @property
    def signal_params(self) -> list[str]:
        return np.array(
            [param for param in self.m.parameters if param.startswith("mu")]
        )

    @property
    def signal_param_indices(self) -> np.ndarray:
        return np.array([self.names.index(param) for param in self.signal_params])

    @property
    def signal_values(self) -> np.ndarray:
        return self.values[self.signal_param_indices]

    @property
    def signal_muarray(self) -> np.ndarray:
        return self.muarray[self.signal_param_indices]
