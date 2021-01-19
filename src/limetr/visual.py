"""
Visualization module
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from scipy.stats import norm
from limetr.core import LimeTr


def funnel(model: LimeTr,
           ax: Axes = None,
           sd_ub: float = 1.0,
           ui_bounds: Tuple[float, float] = None,
           summarize: bool = False) -> Axes:
    """
    Create funnel plot of the model

    Parameters
    ----------
    model : LimeTr
        Given fitting model
    ax : Axes, optional
        Plot axis, by default ``None``. If ``None`` will create new axis.
    sd_ub : float, optional
        Quantile upper bound of total standard deviation, by default 1.
    ui_bounds : Tuple[float, float], optional
        Quantile bounds of uncertainty interval, by default ``None``.
        If ``None`` it will be infered from ``model.inlier_pct``.
    summarize : bool, optional
        If ``summarize``, a text box with information will be printed out,
        default by ``False``.

    Returns
    -------
    Axes
        Axis used to plot the funnel.
    """
    # inputs
    ax = plt.subplots()[1] if ax is None else ax
    if ui_bounds is None:
        ui_bounds = (0.5 - 0.5*model.inlier_pct,
                     0.5 + 0.5*model.inlier_pct)
    color_data = "gray"
    color_funnel = "#77C3E3"
    # compute the residual and total standard error
    soln = model.soln
    r = model.data.obs - model.fevar.mapping(soln["beta"])
    s = np.sqrt(model.data.obs_se**2 +
                np.sum(model.revar.mapping.mat**2*soln["gamma"], axis=1))
    # plot the data
    trim_index = model.data.weight == 0
    ax.scatter(r, s, color=color_data, alpha=0.5)
    ax.scatter(r[trim_index], s[trim_index], color="red", marker="x", alpha=0.5)
    # plot funnel
    s_range = np.array([0.0, np.quantile(s, sd_ub)])
    m_range = norm.ppf(ui_bounds)
    ax.set_ylim(s_range[1], s_range[0])
    for m in m_range:
        ax.plot(s_range*m, s_range, color=color_funnel)
    ax.fill_betweenx(s_range,
                     s_range*m_range[0],
                     s_range*m_range[1], color=color_funnel, alpha=0.3)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
    # add label
    ax.set_xlabel("residual")
    ax.set_ylabel("standard deviation")
    # summarize info
    if summarize:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.95,
                f"num obs={model.data.num_obs}\n"
                f"num groups={model.data.num_groups}\n"
                f"inlier pct={model.inlier_pct:.3f}\n"
                f"num outliers={trim_index.sum():d}\n"
                f"beta={soln['beta']}\n"
                f"gamma={soln['gamma']}\n"
                f"funnel ui=({ui_bounds[0]:.3f}, {ui_bounds[1]:.3f})",
                linespacing=1.5,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=props)

    return ax
