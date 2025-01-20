"""
This file contains data for portfolio creation
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from gurobi_optimods.sharpe_ratio import max_sharpe_ratio
import streamlit as st

# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


### BASE MANIPULATION ###


def portfolio_variance(weights, cov_matrix):
    """From weights and covariance matrix, calculate portfolio variance"""
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def portfolio_std(weights, cov_matrix):
    """From weights and covariance matrix, calculate portfolio standard deviation"""
    return np.sqrt(portfolio_variance(weights, cov_matrix))


### FRONTIER CALCULATIONS ###


## Following optimization function was coauthored with AI (GPT) ##
def _efficient_frontier_opt(
    mean_returns,
    cov_matrix,
    target_returns,
    positive_weights_bound: bool = False,
    min_func=portfolio_std,
) -> list[tuple[float, float]]:
    """
    Formulate and compute Efficient frontier as optimization problem.
    """
    num_assets = len(mean_returns)
    results = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, target=target: np.dot(w, mean_returns) - target,
            },
        )

        # Use following bound for positive weights
        bounds = bounds = [(0, 1)] * num_assets if positive_weights_bound else None
        initial_weights = np.ones(num_assets) / num_assets
        opt_result = minimize(
            min_func,
            initial_weights,
            args=(cov_matrix,),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )
        results.append((target, opt_result.fun))
    return results


def _plot_efficient_frontier_results(
    efficient_results: list[tuple[float, float]],
    plot_individual_stocks: bool = True,
    means: pd.Series = None,
    var: pd.Series = None,
    **kwargs,
    # show_plot: bool = True,
    # plot_results_streamlit: bool = False,
    # return_fig: bool = False,
) -> None:
    """
    Plot efficient frontier (without riskless asset)
    If plot individual stocks is True, also plot the individual stocks
    (sample means and variances must be then provided)
    """

    # Handle kwargs
    show_plot = kwargs.get("show_plot", True)
    plot_results_streamlit = kwargs.get("plot_results_streamlit", False)
    return_fig = kwargs.get("return_fig", False)

    # Unpack the computed optimal values
    opt_means, opt_stds = zip(*efficient_results)

    fig, ax = plt.subplots()
    ax.plot(opt_stds, opt_means, label="Efficient Frontier")
    ax.set_title("Efficient Frontier (Without Risk-Free Asset)")
    ax.set_xlabel("Standard deviation")
    ax.set_ylabel("Mean")
    ax.grid()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(left=-0.001)  # Enforce left starting point of x axis to be 0

    if plot_individual_stocks:
        if means is not None and var is not None:
            stock_stds = np.sqrt(var)

            for index in means.keys():
                index_name = index.split("_")[0]
                ax.scatter([stock_stds[index]], [means[index]], color="green")
                ax.text(
                    stock_stds[index],
                    means[index],
                    index_name,
                    fontsize=9,
                    ha="center",
                    va="bottom",
                )

        else:
            raise ValueError(
                "If plot_individual_stocks is True, means and var must be provided"
            )

    ax.legend()

    if plot_results_streamlit:
        st.pyplot(fig)

    if show_plot:
        plt.show()

    if return_fig:
        return fig, ax

    return None, None


def efficient_frontier_complete(
    means: pd.Series,
    covs: pd.DataFrame,
    var: pd.Series = None,
    plot_results: bool = True,
    **kwargs,
    # positive_weights_bound: bool = False,
    # plot_individual_stocks: bool = True,
    # plot_results_streamlit: bool = False,
    # min_func=portfolio_std,
    # means_max_coef: float = 9,
    # means_min_coef: float = 6,
    # num_points: int = 100,
) -> list[tuple[float, float]]:
    """
    Estimate the set of optimal portfolios and their means and standard deviations
    (efficient frontier). Then plot the results.
    """

    # Handle kwargs
    positive_weights_bound = kwargs.get("positive_weights_bound", False)
    plot_individual_stocks = kwargs.get("plot_individual_stocks", True)
    plot_results_streamlit = kwargs.get("plot_results_streamlit", False)
    min_func = kwargs.get("min_func", portfolio_std)
    means_max_coef = kwargs.get("means_max_coef", 9)
    means_min_coef = kwargs.get("means_min_coef", 6)
    num_points = kwargs.get("num_points", 100)

    # Define a range of optimal mean values for which to solve
    target_returns = np.linspace(
        means.min() * means_min_coef, means.max() * means_max_coef, num_points
    )

    # Compute (Estimate) the efficient frontier
    efficient_results = _efficient_frontier_opt(
        mean_returns=means,
        cov_matrix=covs,
        target_returns=target_returns,
        positive_weights_bound=positive_weights_bound,
        min_func=min_func,
    )

    # Plot the efficient frontier
    if plot_results or plot_results_streamlit:
        _plot_efficient_frontier_results(
            efficient_results,
            plot_individual_stocks=plot_individual_stocks,
            means=means,
            var=var,
            show_plot=plot_results,
            plot_results_streamlit=plot_results_streamlit,
        )

    return efficient_results


def _compute_mark_helper_quantities(means: pd.Series, covs: pd.DataFrame):
    """
    Compute helper quantities for the Markowitz efficient frontier
    """
    inv_cov = np.linalg.inv(covs)
    unit_vec = np.ones_like(means)

    # Helper quentities for nice formulas
    a_quant = unit_vec.T @ (inv_cov @ unit_vec)
    b_quant = unit_vec.T @ (inv_cov @ means)
    c_quant = means.T @ (inv_cov @ means)

    return inv_cov, unit_vec, a_quant, b_quant, c_quant


def _compute_tangent_portfolio(
    means: pd.Series, covs: pd.DataFrame, riskless_rate: float
):
    """
    Compute the tangent portfolio
    """
    inv_cov, unit_vec, a_quant, b_quant, _ = _compute_mark_helper_quantities(
        means, covs
    )

    # Formula for weights of tangent portfolio (viz. Analýza Investic)
    # viz https://www.karlin.mff.cuni.cz/~rusy/AnalyzaInvestic/InvestmentAnalysisNotes.pdf
    weights_tangent = (inv_cov @ (means - (unit_vec * riskless_rate))) / (
        b_quant - riskless_rate * a_quant
    )

    mean_tangent = np.dot(means, weights_tangent)

    std_tangent = np.sqrt(weights_tangent.T @ (covs @ weights_tangent))

    return weights_tangent, mean_tangent, std_tangent


def _plot_capital_market_line(
    riskless_rate: float,
    mean_tangent: float,
    std_tangent: float,
    mean_tangent_mul: float = 1.5,
    show_plot: bool = False,
):
    """
    Plot the Capital Market Line
    """
    # Plot the yield of a riskfree asst
    plt.scatter(0, riskless_rate, color="purple", label="Risk-Free Asset", zorder=3)

    ##Capital Market line
    # Define set of x values (stds)
    cml_x = np.linspace(0, std_tangent * mean_tangent_mul, 100)
    # Compute corresponding y values (means)
    cml_y = riskless_rate + (cml_x / std_tangent) * (mean_tangent - riskless_rate)

    plt.plot(
        cml_x,
        cml_y,
        color="orange",
        linestyle="--",
        label="Capital Market Line",
        zorder=2,
    )

    if show_plot:
        plt.show()


def _plot_efficient_frontier_riskless(
    efficient_results: list[tuple[float, float]],
    means: pd.Series,
    var: pd.Series,
    riskless_rate: float,
    **kwargs,
    # plot_individual_stocks: bool = True,
    # plot_results: bool = True,
    # mean_tangent: float = None,
    # std_tangent: float = None,
    # mean_tangent_mul: float = 1.5,
    # plot_results_streamlit: bool = False,
):
    """
    Plot efficient frontier with a riskless asset
    along with tangent portfolio and capital market line
    """

    # Handle kwargs
    plot_individual_stocks = kwargs.get("plot_individual_stocks", True)
    plot_results = kwargs.get("plot_results", True)
    mean_tangent = kwargs.get("mean_tangent", None)
    std_tangent = kwargs.get("std_tangent", None)
    mean_tangent_mul = kwargs.get("mean_tangent_mul", 1.5)
    plot_results_streamlit = kwargs.get("plot_results_streamlit", False)
    right_x_lim = kwargs.get("right_x_lim", None)

    # Plot the efficient frontier without riskless asset
    fig, ax = _plot_efficient_frontier_results(
        efficient_results,
        plot_individual_stocks,
        means,
        var,
        show_plot=False,
        return_fig=True,
    )

    # Plot tangent portfolio
    if mean_tangent is not None and std_tangent is not None:
        ax.scatter(
            [std_tangent],
            [mean_tangent],
            color="red",
            label="Tangent Portfolio",
            zorder=3,
        )

    # Plot Capital Market line
    _plot_capital_market_line(
        riskless_rate, mean_tangent, std_tangent, mean_tangent_mul, show_plot=False
    )

    ax.set_title("Efficient Frontier (With Risk-Free Asset)")
    ax.set_xlabel("Standard deviation")
    ax.set_ylabel("Mean")
    ax.set_xlim(left=-0.001)
    if right_x_lim is not None:
        ax.set_xlim(right=right_x_lim)
    ax.legend()
    ax.grid()

    if plot_results_streamlit:
        st.pyplot(fig)

    if plot_results:
        plt.show()


def efficient_frontier_complete_riskless(
    means: pd.Series,
    covs: pd.DataFrame,
    var: pd.Series,
    riskless_rate: float,
    **kwargs,
    # plot_individual_stocks: bool = True,
    # plot_results: bool = True,
    # mean_tangent_mult: float = 1.5,
    # plot_results_streamlit: bool = False,
):
    """
    Compute the efficient frontier with a riskless asset
    """

    # Handle kwargs
    plot_individual_stocks = kwargs.get("plot_individual_stocks", True)
    plot_results = kwargs.get("plot_results", True)
    mean_tangent_mult = kwargs.get("mean_tangent_mult", 1.5)
    plot_results_streamlit = kwargs.get("plot_results_streamlit", False)
    right_x_lim = kwargs.get("right_x_lim", None)

    efficient_results = efficient_frontier_complete(
        means, covs, var, plot_results=False
    )

    weights_tangent, mean_tangent, std_tangent = _compute_tangent_portfolio(
        means, covs, riskless_rate
    )

    _plot_efficient_frontier_riskless(
        efficient_results,
        means,
        var,
        riskless_rate,
        plot_individual_stocks=plot_individual_stocks,
        plot_results=plot_results,
        mean_tangent=mean_tangent,
        std_tangent=std_tangent,
        mean_tangent_mul=mean_tangent_mult,
        plot_results_streamlit=plot_results_streamlit,
        right_x_lim=right_x_lim,
    )

    return weights_tangent, mean_tangent, std_tangent


### MARKOWITZ PORTFOLIO ###


def compute_markowitz_weights(means, covs, riskless_rate, required_return):
    """
    Compute Markowitz portfolio weights based on required return and riskless rate
    """

    inv_cov, unit_vec, a_quant, b_quant, c_quant = _compute_mark_helper_quantities(
        means, covs
    )

    # Formulas are taken from Analýza Investic (TODO: Add reference)
    nominator = (required_return - riskless_rate) * (
        inv_cov @ (means - (riskless_rate * means))
    )
    denominator = (
        c_quant - (2 * b_quant * riskless_rate) + (a_quant * (riskless_rate**2))
    )

    shares_weights = nominator / denominator

    riskless_asset_weight = 1 - np.dot(shares_weights, unit_vec)

    return shares_weights, riskless_asset_weight


### PORTFOLIO MANIPULATION ###


def combine_pure_strat_with_riskless(
    weigth_comp_func, weight_comp_kwargs: dict, expected_riskless_asset_weight: float
):
    """
    Combine a pure strategy with a riskless asset
    Expects a function that returns only the weights of shares.
    Adds certain part of the weight to the riskless asset and adjusts the share weights accordingly.
    """
    # Compute the weights of the pure strategy
    pure_weights = weigth_comp_func(**weight_comp_kwargs)
    share_weights = pure_weights * (1 - expected_riskless_asset_weight)

    return share_weights, expected_riskless_asset_weight


def combine_share_weights_with_riskless(
    share_weights: pd.Series, expected_riskless_asset_weight: float
):
    """
    Combine share weights with a riskless asset.
    Expect a series of share weights and a weight of the riskless asset.
    """
    adjusted_share_weights = share_weights * (1 - expected_riskless_asset_weight)

    return adjusted_share_weights, expected_riskless_asset_weight


### NON-MARKOWITZ PORTFOLIOS ###


def compute_equal_weights_portfolio(means: pd.Series, use_riskless_asset: bool = False):
    """
    Compute equal weights portfolio
    """
    # Calculate number of assets among which to distribute the weights
    asset_num = len(means) + 1 if use_riskless_asset else len(means)
    equal_weight = 1 / asset_num
    riskless_asset_weight = equal_weight if use_riskless_asset else 0
    shares_weights = np.ones_like(means) * equal_weight

    return shares_weights, riskless_asset_weight


def compute_sharpe_nonnegative_weights_external(
    means: pd.Series, covs: pd.DataFrame, riskless_rate: float
):
    """
    Compute the sharpe ratio maximizing portfolio with non-negative weights
    Functionality imported from the gurobi_optimods package
    """
    # pylint: disable=missing-kwoa
    sharpe_share_results = max_sharpe_ratio(covs, means, rf_rate=riskless_rate)
    sharpe_share_weights = sharpe_share_results.x

    return sharpe_share_weights


def compute_max_sharpe_weights(
    means: pd.Series, covs: pd.DataFrame, riskless_rate: float
):
    """
    Compute the sharpe ratio maximizing portfolio without bounding conditions
    The max sharpe weights are equivalent to the tangency portfolio
    Derivation can be found eg. in
    https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
    """
    inv_cov = np.linalg.inv(covs)
    riskless_adjusted_means = means - riskless_rate
    unit_vec = np.ones_like(means)

    nominator = inv_cov @ riskless_adjusted_means
    denominator = unit_vec @ inv_cov @ riskless_adjusted_means

    sharpe_weights = nominator / denominator

    return sharpe_weights


def compute_inverse_volatility_weighed_portfolio(var: pd.DataFrame):
    """
    Compute the inverse volatility weighted portfolio weights
    """
    inv_vol = 1 / np.sqrt(var)
    inv_vol_weights = inv_vol / inv_vol.sum()

    return inv_vol_weights


def compute_minimum_variance_portfolio(means: pd.Series, covs: pd.DataFrame):
    """
    Compute the minimum variance portfolio weights
    """
    inv_cov = np.linalg.inv(covs)
    unit_vec = np.ones_like(means)

    nominator = inv_cov @ unit_vec
    denominator = unit_vec @ inv_cov @ unit_vec

    min_var_weights = nominator / denominator

    return min_var_weights
