import os
import sys
from typing import Any, Callable

import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties as unc
from tqdm import tqdm
from uncertainties import unumpy as unp

from utils.fit import Fit


def get_values(
    df: pd.DataFrame, var: str, edges: np.ndarray, query: str | None = None
) -> np.ndarray:
    """Fill histogram and return values with stat. uncertainty."""
    if query is not None:
        df = df.query(query)
    h = hist.Hist(hist.axis.Variable(edges))
    h.fill(df[var])
    return unp.uarray(h.values(), np.sqrt(h.variances()))


def get_efficiency(
    df: pd.DataFrame, var: str, edges: np.ndarray, query: str
) -> np.ndarray:
    """Calculate efficiencies with binomial uncertainties and allow for empty bins."""

    h_num = hist.Hist(hist.axis.Variable(edges))
    h_num.fill(df.query(query)[var])
    num = h_num.values()

    h_denom = hist.Hist(hist.axis.Variable(edges))
    h_denom.fill(df[var])
    denom = h_denom.values()

    eff = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    unc = np.sqrt(
        np.divide(eff * (1 - eff), denom, out=np.zeros_like(num), where=denom != 0)
    )
    return unp.uarray(eff, unc)


def get_migration_matrix(
    df: pd.DataFrame,
    truth_var: str,
    reco_var: str,
    edges: np.ndarray,
    query: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Calcualte migration matrix to unfold reco_var to truth_var."""
    if query is not None:
        df = df.query(query)

    mig_hist = hist.Hist(hist.axis.Variable(edges), hist.axis.Variable(edges))
    mig_hist.fill(df[truth_var], df[reco_var])

    mig = unp.uarray(mig_hist.values(), np.sqrt(mig_hist.variances()))
    # Histogram stores first axis as column (y), and second axis as row (x).
    # Transpose, so that first axis corresponds to x-axis.
    mig = np.transpose(mig)
    # Normalize so taht sum of truth values is one.
    if normalize:
        num = mig
        denom = mig.sum(axis=0)[np.newaxis, :]
        mig = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return mig


def plot_values(
    values: list[np.ndarray],
    edges: np.ndarray,
    edge_value: np.ndarray | None = None,
    labels: list[str] | None = None,
    histtypes: list[str] | None = None,
    markers: list[str] | None = None,
    markersize: int = 2,
    xerr: bool | np.ndarray = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = r"\# Muons [/cm$^2$/fb$^{-1}$]",
    ylim: tuple[float, float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    **kwargs,
):
    size = len(values)
    if histtypes is None:
        histtypes = ["step"] * size
    if labels is None:
        labels = [""] * size
    if markers is None:
        markers = ["."] * size
    fig, ax = plt.subplots(figsize=(6, 4))
    for _values, _label, _histtype, _marker in zip(values, labels, histtypes, markers):
        mplhep.histplot(
            H=unp.nominal_values(_values),
            bins=edges,
            xerr=xerr,
            yerr=unp.std_devs(_values),
            histtype=_histtype,
            marker=_marker,
            markersize=markersize,
            label=_label,
            ax=ax,
            flow=None,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(edges[0], edges[-1])
    if edge_value is not None:
        set_xticks(edge_value, integer_edges=True, ax=ax, fraction=True, **kwargs)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, None)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    plt.legend()
    plt.show()


def plot_efficiency(
    efficiency: np.ndarray,
    edges: np.ndarray,
    xlabel: str = "",
    ylabel: str = "Efficiency",
) -> None:
    if isinstance(efficiency[0], unc.core.Variable):
        eff = unp.nominal_values(efficiency)
        eff_unc = unp.std_devs(efficiency)
    elif isinstance(efficiency[0], np.number):
        eff = efficiency
        eff_unc = None
    else:
        raise ValueError(f"Type {type(efficiency[0])} not implemented.")
    fig, ax = plt.subplots(figsize=(6, 4))
    mplhep.histplot(
        H=eff, bins=edges, xerr=True, yerr=eff_unc, histtype="errorbar", ax=ax
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(edges[0], edges[-1])
    plt.tight_layout()
    plt.show()


def get_uncertain_annotation_formatter(
    write_unc: bool = False, decimals: int = 3
) -> Callable[[Any], str]:
    if write_unc:

        def _fmt_func(x):
            fmt_string = f"   {{:.{decimals}f}}\n$\\pm${{:.{decimals}f}}"
            return fmt_string.format(unc.nominal_value(x), unc.std_dev(x))

        return _fmt_func
    else:

        def _fmt_func(x):
            fmt_string = f"{{:.{decimals}f}}"
            return fmt_string.format(unc.nominal_value(x))

        return _fmt_func


def get_str_fraction(value: float) -> str:
    """Return value as string in the form of sign * 1/abs(value)"""
    sign = "" if value > 0 else "-"
    return f"{sign}1/{int(1 / abs(value))}"


def set_xticks(
    edges: np.ndarray,
    ax: plt.Axes,
    fontsize: int = 13,
    rotation: int = 30,
    integer_edges: bool = True,
    fraction: bool = False,
) -> None:
    if fraction:
        labels = [get_str_fraction(value) for value in edges]
    else:
        labels = [str(edge) for edge in edges]
    if integer_edges:
        ax.set_xticks(np.arange(len(edges)))
    else:
        ax.set_xticks(edges)
    ax.set_xticklabels(labels, rotation=rotation, fontdict={"size": fontsize})


def set_yticks(
    edges: np.ndarray,
    ax: plt.Axes,
    rotation: int = 0,
    integer_edges: bool = True,
    fraction: bool = False,
) -> None:
    if fraction:
        labels = [get_str_fraction(value) for value in edges]
    else:
        labels = [str(edge) for edge in edges]
    if integer_edges:
        ax.set_yticks(np.arange(len(edges)))
    else:
        ax.set_yticks(edges)
    ax.set_yticklabels(labels, rotation=rotation)


def plot_migration_matrix(
    migration_matrix: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    norm: str = "linear",
    xedges: np.ndarray | None = None,
    yedges: np.ndarray | None = None,
    fontsize: int = 9,
    fraction: bool = False,
    figsize: tuple[float, float] = (6, 6),
    out_path: str | None = None,
) -> None:
    fmt_func = get_uncertain_annotation_formatter(write_unc=False, decimals=2)
    _plot_style_dict = {
        "annot": pd.DataFrame(migration_matrix).map(fmt_func),
        "fmt": "s",
        "cmap": "Blues",
        "vmin": 0,
        "vmax": 1,
        "square": True,
        "cbar_kws": {"fraction": 0.046, "pad": 0.04},
        "linewidths": 0.5,
        "linecolor": "k",
        "annot_kws": {"size": fontsize},
    }
    nominal_matrix = unp.nominal_values(migration_matrix)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(nominal_matrix, ax=ax, norm=norm, **_plot_style_dict)
    ax.set_aspect("equal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xedges is not None:
        set_xticks(xedges, fraction=fraction, ax=ax, rotation=90)
    if yedges is not None:
        set_yticks(yedges, fraction=fraction, ax=ax)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()


def create_unique_event_id(event_id: np.ndarray, offset: int = 5000) -> np.ndarray:
    """
    The sample consists of several chunks with event ids between 0 and 5000.
    Therefore, the combined event id is not monotonic increasing / unique.
    To get a unique event id we add offset (5000) for each new chunk.
    """
    ids_jump = np.argwhere(np.diff(event_id) < 0)
    for id in ids_jump.flatten():
        event_id[id + 1 :] += offset
    return event_id


def rho_scan(
    meas: np.ndarray,
    initial_guess: dict,
    mig: np.ndarray,
    mig_unc: np.ndarray | None = None,
    eff: np.ndarray | None = None,
    eff_unc: np.ndarray | None = None,
    reg_values: np.ndarray | None = None,
    reg_type: str = "log_curvature",
) -> dict:
    """
    Scan regularization values and find the optimal tau by minimizing the average global
    correlation coefficient rho across signal bins.

    For each tau the global correlation coefficient per bin is computed from the
    post-fit covariance matrix C of the signal parameters:

        rho_i = sqrt(1 - 1 / (C^{-1}_{ii} * C_{ii}))
    """
    if reg_values is None:
        reg_values = np.logspace(-2, 6, 25)

    rho_values = []

    for i, reg in tqdm(enumerate(reg_values)):
        fit = Fit(
            meas=meas,
            initial_guess=initial_guess,
            mig=mig,
            mig_unc=mig_unc,
            eff=eff,
            eff_unc=eff_unc,
            regularization=float(reg),
            reg_type=reg_type,
        )

        # Signal-parameter covariance matrix
        sig_idx = fit.signal_param_indices
        cov_full = np.array(fit.covariance)
        cov = cov_full[np.ix_(sig_idx, sig_idx)]

        # Global correlation coefficient per bin
        try:
            cov_inv = np.linalg.inv(cov)
            rhos = np.sqrt(
                np.maximum(1.0 - 1.0 / (np.diag(cov_inv) * np.diag(cov)), 0.0)
            )
            rho_values.append(np.mean(rhos))
        except np.linalg.LinAlgError:
            rho_values.append(np.nan)

    min_idx = np.argmin(rho_values)
    opt_reg = reg_values[min_idx]

    return {"reg_values": reg_values, "rho_values": rho_values, "opt_reg": opt_reg}


def plot_rho_scan(reg_values: np.ndarray, rho_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(reg_values, rho_values, "o-")
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization strength $\tau$")
    ax.set_ylabel(r"Correlation $\rho$")
    plt.tight_layout()
    plt.show()


def setup_env() -> None:
    # Setup ROOT path
    sys.path.insert(0, "/opt/homebrew/lib/root")
    os.environ["ROOTSYS"] = "/opt/homebrew"
    import ROOT


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "errorbar.capsize": 2,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


# def get_weighted_mean_std(
#     values: np.ndarray, weights: np.ndarray
# ) -> tuple[float, float]:
#     average = np.average(values, weights=weights)
#     variance = np.average((values - average) ** 2, weights=weights)
#     return average, np.sqrt(variance)


# def get_migration_matrix_from_hist(migration_hist: hist.Hist) -> np.ndarray:
#     migration_uarray = unp.uarray(
#         migration_hist.values(), np.sqrt(migration_hist.variances())
#     )
#     # Histogram stores first axis as column (y), and second axis as row (x).
#     # Transpose, so that first axis corresponds to x-axis.
#     migration_uarray = np.transpose(migration_uarray)
#     num = migration_uarray
#     denom = migration_uarray.sum(axis=0)[np.newaxis, :]
#     return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
