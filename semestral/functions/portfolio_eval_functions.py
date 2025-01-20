"""
Functions for portfolio evaluation
"""

# pylint: disable=too-many-positional-arguments, too-many-locals
# Majority of the functions were rewritten to obide by pylint
# 2 of them were kept as is for better readability

import sys
import os
from typing import Literal
import inspect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801

# Workaround wierd python relative import behavior and pylint warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from functions.data_exploration_functions import get_descriptive_statistics
    from functions.portfolio_functions import (
        combine_share_weights_with_riskless,
        compute_markowitz_weights,
        compute_minimum_variance_portfolio,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def periodical_gain_for_portfolio(
    portfolio_value: float,
    shares_weights: pd.Series,
    # riskfree_asset_weight: float,
    periodical_share_returns: pd.Series,
    # periodical_riskfree_return: float,
    # add_one: bool = True,
    **kwargs,
) -> float:
    """
    Single period portfolio value incrementation
    """

    # Handle kwargs
    riskfree_asset_weight = kwargs.get("riskfree_asset_weight", 0.0)
    periodical_riskfree_return = kwargs.get("periodical_riskfree_return", 0.0)
    add_one = kwargs.get("add_one", True)

    # Returns are expected to be in raw multiplicative form
    if add_one:
        periodical_share_returns += 1
        periodical_riskfree_return += 1
    # Increment the portfolio value
    shares_capital_incremented = shares_weights * periodical_share_returns
    riskfree_capital_incremented = riskfree_asset_weight * periodical_riskfree_return

    return (
        np.sum(shares_capital_incremented) + riskfree_capital_incremented
    ) * portfolio_value


## Following helper function was suggested by GPT
def _filter_kwargs(func, kwargs):
    """
    Helper function for filtering out invalid keyword arguments for a function
    """
    valid_keys = set(inspect.signature(func).parameters.keys())
    return {key: value for key, value in kwargs.items() if key in valid_keys}


def _get_daily_invested_weights(
    portfolio_comp_func,
    stat_kwargs: dict,
    additional_kwargs: dict = None,
    combine_pure_strat_with_riskless_coef: float = 0,
) -> tuple:
    """
    Pass correct kwargs to the function. Get daily share weights and riskless asset weight
    """
    # Combine the two sets of kwargs into one
    combined_kwargs = {**stat_kwargs, **(additional_kwargs or {})}

    # Filter out invalid kwargs
    valid_kwargs = _filter_kwargs(portfolio_comp_func, combined_kwargs)

    weights_result = portfolio_comp_func(**valid_kwargs)

    # If two values are returned, assume that the first one is the stock weights
    # and the second one is the riskless asset weight
    if isinstance(weights_result, tuple) and len(weights_result) == 2:
        return weights_result[0], weights_result[1]

    # If single value is returned, assume that it is the stock weights (should be of type pd.Series)
    if isinstance(weights_result, (pd.Series, np.ndarray)):
        return combine_share_weights_with_riskless(
            share_weights=weights_result,
            expected_riskless_asset_weight=combine_pure_strat_with_riskless_coef,
        )
    raise ValueError("Unexpected return value from the portfolio computation function")


def single_daily_reinvesting_strategy(
    initial_portfolio_value: float,
    full_data: pd.DataFrame,
    today_date: str,
    start_est_date: str,
    # calc_share_returns_regexp=r"interdaily_pct_change$",
    # physical_share_returns_regexp=r"interdaily_pct_change$",
    # riskfree_rate_col: str = "daily_riskless_yield",
    # weights_computation_function=compute_markowitz_weights,
    # reuse_previous_weights: bool = False,
    # previous_stock_weights: pd.Series = None,
    # previous_riskfree_weight: float = None,
    # weight_comp_kwargs: dict = None,
    # combine_pure_strat_with_riskless_coef: float = 0.0,
    **kwargs,
):
    """
    For a single day:
        - Aquire data available, compute statistics
        - Compute the daily portfolio
        - Trade according to the used portfolio
    """

    # Handle kwargs
    calc_share_returns_regexp = kwargs.get(
        "calc_share_returns_regexp", r"interdaily_pct_change$"
    )
    physical_share_returns_regexp = kwargs.get(
        "physical_share_returns_regexp", r"interdaily_pct_change$"
    )
    riskfree_rate_col = kwargs.get("riskfree_rate_col", "daily_riskless_yield")
    weights_computation_function = kwargs.get(
        "weights_computation_function", compute_markowitz_weights
    )
    reuse_previous_weights = kwargs.get("reuse_previous_weights", False)
    previous_stock_weights = kwargs.get("previous_stock_weights", None)
    previous_riskfree_weight = kwargs.get("previous_riskfree_weight", None)
    weight_comp_kwargs = kwargs.get("weight_comp_kwargs", None)
    combine_pure_strat_with_riskless_coef = kwargs.get(
        "combine_pure_strat_with_riskless_coef", 0.0
    )

    # Get daily returns
    today_riskfree_return = full_data[riskfree_rate_col].loc[today_date]
    today_share_returns = full_data.filter(regex=physical_share_returns_regexp).loc[
        today_date
    ]

    if reuse_previous_weights:
        # Check whether previous weights are provided
        if previous_stock_weights is None or previous_riskfree_weight is None:
            raise ValueError(
                "Previous weights must be provided if reuse_previous_weights set to True"
            )
        # Keep weights from previous day
        today_share_weights = previous_stock_weights
        today_riskfree_weight = previous_riskfree_weight
    else:
        # Find the immediate previous date
        previous_date = full_data.index[full_data.index < today_date].max()

        # Compute daily stats (Only use data up to the previous date)
        today_means, today_var, today_covs, today_corr = get_descriptive_statistics(
            full_data,
            print_stats=False,
            start_date=start_est_date,
            end_date=previous_date,
            col_regexp_substring=calc_share_returns_regexp,
        )

        # Convert daily stats to dictionary to-be-used as kwargs
        # for the weights computation function
        # Make sure that the keys are named the same as the function arguments
        today_stats = {
            "means": today_means,
            "covs": today_covs,
            "var": today_var,
            "corr": today_corr,
            "riskless_rate": today_riskfree_return,
        }

        # Get today's weights
        today_share_weights, today_riskfree_weight = _get_daily_invested_weights(
            weights_computation_function,
            today_stats,
            weight_comp_kwargs,
            combine_pure_strat_with_riskless_coef,
        )

    # Compute the incrementation of the portfolio value
    incremented_portfolio_value = periodical_gain_for_portfolio(
        # initial_portfolio_value,
        # today_share_weights,
        # today_riskfree_weight,
        # today_share_returns,
        # today_riskfree_return,
        portfolio_value=initial_portfolio_value,
        shares_weights=today_share_weights,
        periodical_share_returns=today_share_returns,
        periodical_riskfree_return=today_riskfree_return,
        riskfree_asset_weight=today_riskfree_weight,
    )

    return incremented_portfolio_value, today_share_weights, today_riskfree_weight


def _get_potential_investment_dates(
    start_trade_date: str | pd.Timestamp, end_trade_date: str | pd.Timestamp
):
    # Enforce dates in pd.Timestamp format
    start_trade_date = pd.Timestamp(start_trade_date)
    end_trade_date = pd.Timestamp(end_trade_date)
    # Calculate the potential investment dates
    potential_investment_dates = pd.date_range(
        start=start_trade_date, end=end_trade_date
    )
    return potential_investment_dates


def _get_first_day_weights(
    full_data: pd.DataFrame,
    start_est_date: str | pd.Timestamp,
    today_date: str | pd.Timestamp,
    portfolio_comp_func,
    **kwargs,
    # calc_share_returns_regexp: str = r"interdaily_pct_change$",
    # weight_computation_kwargs: dict = None,
    # combine_pure_strat_with_riskless_coef: float = 0,
):
    # Handle kwargs
    calc_share_returns_regexp = kwargs.get(
        "calc_share_returns_regexp", r"interdaily_pct_change$"
    )
    weight_computation_kwargs = kwargs.get("weight_computation_kwargs", None)
    combine_pure_strat_with_riskless_coef = kwargs.get(
        "combine_pure_strat_with_riskless_coef", 0.0
    )

    # Find the immediate previous date
    previous_date = full_data.index[full_data.index < today_date].max()

    # Compute daily stats (Only use data up to the previous date)
    first_date_stats_kwargs = get_descriptive_statistics(
        full_data,
        print_stats=False,
        start_date=start_est_date,
        end_date=previous_date,
        col_regexp_substring=calc_share_returns_regexp,
    )

    # Get today's weights
    today_share_weights, today_riskfree_weight = _get_daily_invested_weights(
        portfolio_comp_func=portfolio_comp_func,
        stat_kwargs=first_date_stats_kwargs,
        additional_kwargs=weight_computation_kwargs,
        combine_pure_strat_with_riskless_coef=combine_pure_strat_with_riskless_coef,
    )

    return today_share_weights, today_riskfree_weight


def _recalculate_fixed_portfolio_weights(
    full_data: pd.DataFrame,
    today_date: str | pd.Timestamp,
    today_share_weights: pd.Series,
    today_riskfree_weight: float,
    **kwargs,
    # riskfree_rate_col: str = "daily_riskless_yield",
    # physical_share_returns_regexp: str = r"interdaily_pct_change$",
    # add_one: bool = True,
):
    """
    Update portfolio weights accordingly to the market shifts (so that the portfolio remains fixed)
    """

    # Handle kwargs
    riskfree_rate_col = kwargs.get("riskfree_rate_col", "daily_riskless_yield")
    physical_share_returns_regexp = kwargs.get(
        "physical_share_returns_regexp", r"interdaily_pct_change$"
    )
    add_one = kwargs.get("add_one", True)

    # Get daily returns
    today_riskfree_return = full_data[riskfree_rate_col].loc[today_date]
    today_share_returns = full_data.filter(regex=physical_share_returns_regexp).loc[
        today_date
    ]
    if add_one:
        today_share_returns += 1
        today_riskfree_return += 1

    # Compute the updated weights
    denominator = (
        np.sum(today_share_weights * today_share_returns)
        + today_riskfree_weight * today_riskfree_return
    )
    today_share_weights = today_share_weights * today_share_returns / denominator
    today_riskfree_weight = today_riskfree_weight * today_riskfree_return / denominator

    return today_share_weights, today_riskfree_weight


# Following function was kept as is, rather than kwargs refactoring for better readability
# pylint: disable=too-many-arguments,too-many-locals
def multiple_daily_reinvesting_strategy(
    initial_portfolio_value: float,
    full_data: pd.DataFrame,
    start_est_date: str | pd.Timestamp,
    start_trade_date: str | pd.Timestamp,
    end_trade_date: str | pd.Timestamp,
    wieghts_recalculation_style: Literal[
        "daily", "fixed_weights", "fixed_portfolio"
    ] = "daily",
    calc_share_returns_regexp: str = r"interdaily_pct_change$",
    physical_share_returns_regexp: str = r"interdaily_pct_change$",
    riskfree_rate_col: str = "daily_riskless_yield",
    weights_computation_function=compute_minimum_variance_portfolio,
    weight_computation_kwargs: dict = None,
    combine_pure_strat_with_riskless_coef: float = 0,
):
    """
    Apply an investing strategy over a period of time
    """

    if wieghts_recalculation_style not in ["daily", "fixed_weights", "fixed_portfolio"]:
        raise ValueError("Invalid wieghts_recalculation_style value")

    # Initialize the arrays
    daily_portfolio_values = []
    daily_portfolio_share_weights = []
    daily_portfolio_riskfree_weights = []
    realized_trade_dates = []

    # Get the potential investment dates
    potential_investment_dates = _get_potential_investment_dates(
        start_trade_date, end_trade_date
    )

    # Initialize the portfolio value
    portfolio_value = initial_portfolio_value

    reuse_previous_weights = False

    ## Handle initial trade for fixed weights or fixed portfolio separately
    today_share_weights, today_riskfree_weight = None, None

    # Iterate over the potential investment dates
    for today_date in potential_investment_dates:
        if today_date in full_data.index:

            incremented_portfolio_value, today_share_weights, today_riskfree_weight = (
                single_daily_reinvesting_strategy(
                    initial_portfolio_value=portfolio_value,
                    full_data=full_data,
                    today_date=today_date,
                    start_est_date=start_est_date,
                    calc_share_returns_regexp=calc_share_returns_regexp,
                    physical_share_returns_regexp=physical_share_returns_regexp,
                    riskfree_rate_col=riskfree_rate_col,
                    weights_computation_function=weights_computation_function,
                    reuse_previous_weights=reuse_previous_weights,
                    previous_stock_weights=today_share_weights,
                    previous_riskfree_weight=today_riskfree_weight,
                    weight_comp_kwargs=weight_computation_kwargs,
                    combine_pure_strat_with_riskless_coef=combine_pure_strat_with_riskless_coef,
                )
            )

            # Append the results
            portfolio_value = incremented_portfolio_value
            realized_trade_dates.append(today_date)
            daily_portfolio_values.append(incremented_portfolio_value)
            daily_portfolio_share_weights.append(today_share_weights)
            daily_portfolio_riskfree_weights.append(today_riskfree_weight)

            # From now on, reuse the previous weights
            # for "fixed_weights" and "fixed_portfolio" strategies
            # reuse_previous_weights = (
            #     False if wieghts_recalculation_style == "daily" else True
            # )

            reuse_previous_weights = wieghts_recalculation_style != "daily"

            if wieghts_recalculation_style == "fixed_portfolio":
                today_share_weights, today_riskfree_weight = (
                    _recalculate_fixed_portfolio_weights(
                        full_data=full_data,
                        today_date=today_date,
                        today_share_weights=today_share_weights,
                        today_riskfree_weight=today_riskfree_weight,
                        riskfree_rate_col=riskfree_rate_col,
                        physical_share_returns_regexp=physical_share_returns_regexp,
                        add_one=True,
                    )
                )

    return (
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
    )


def convert_multiple_reinvestment_output_to_df(
    daily_portfolio_values: list[float],
    daily_portfolio_share_weights: list[pd.Series],
    daily_portfolio_riskfree_weights: list[float],
    realized_trade_dates: list[pd.Timestamp],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Convert output of multiple_daily_reinvesting_strategy to a pandas df indexed by dates.
    """
    # Convert the lists to DataFrame
    daily_portfolio_values_df = pd.DataFrame(
        daily_portfolio_values, index=realized_trade_dates, columns=["portfolio_value"]
    )
    daily_portfolio_riskfree_weights_df = pd.DataFrame(
        daily_portfolio_riskfree_weights,
        index=realized_trade_dates,
        columns=["riskfree_weight"],
    )
    daily_portfolio_share_weights_df = pd.DataFrame(
        daily_portfolio_share_weights,
        index=realized_trade_dates,
        columns=[f"{ticker}_weight" for ticker in tickers],
    )

    # Combine the three DataFrames
    comb_df = pd.concat(
        [
            daily_portfolio_values_df,
            daily_portfolio_riskfree_weights_df,
            daily_portfolio_share_weights_df,
        ],
        axis=1,
    )

    return comb_df


def plot_multiple_reinvestment_results(
    output_df: pd.DataFrame,
    plot_portfolio_value: bool = True,
    plot_weights: bool = True,
    show_plots: bool = True,
    show_plots_streamlit: bool = False,
):
    """
    Plot the results of multiple_daily_reinvesting_strategy
    Requires the output to be in pandas DataFrame format
    (Use convert_multiple_reinvestment_output_to_df function)
    """
    # Plot the portfolio value over time
    if plot_portfolio_value:
        if "portfolio_value" not in output_df.columns:
            raise ValueError("Wrong DataFrame format. Portfolio value column not found")
        fig, ax = plt.subplots()
        ax.plot(output_df["portfolio_value"])
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid()

        if show_plots_streamlit:
            st.pyplot(fig)

        if show_plots:
            plt.show()

    if plot_weights:
        # Plot the weights over time
        weights_cols = [col for col in output_df.columns if "_weight" in col]
        if len(weights_cols) == 0:
            raise ValueError("Wrong DataFrame format. Weights columns not found")
        fig, ax = plt.subplots()
        output_df[weights_cols].plot(ax=ax)
        ax.set_title("Portfolio Weights Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.grid()

        if show_plots_streamlit:
            st.pyplot(fig)

        if show_plots:
            plt.show()
