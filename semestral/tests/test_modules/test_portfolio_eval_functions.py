"""
Tests for the portfolio_eval_functions file.
"""

import sys
import os
from unittest.mock import patch

# import pytest


import matplotlib
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.portfolio_eval_functions import (
        periodical_gain_for_portfolio,
        single_daily_reinvesting_strategy,
        multiple_daily_reinvesting_strategy,
        convert_multiple_reinvestment_output_to_df,
        plot_multiple_reinvestment_results,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Block plots from showing up
matplotlib.use("Agg")


def test_periodical_gain_for_portfolio(simple_weights, simple_share_returns):
    """
    Test for the periodical_gain_for_portfolio function
    """
    incremented_value = periodical_gain_for_portfolio(
        portfolio_value=1000,
        shares_weights=simple_weights - 0.05,
        periodical_share_returns=simple_share_returns,
        riskfree_asset_weight=0.1,
        periodical_riskfree_return=0.001,
        add_one=True,
    )

    assert isinstance(incremented_value, float)
    assert round(incremented_value, 2) == 1130.60


def test_single_daily_reinvesting_strategy(base_comb_df, base_ticker_set):
    """Test for the single_daily_reinvesting_strategy function"""
    incremented_portfolio_value, today_share_weights, today_riskfree_weight = (
        single_daily_reinvesting_strategy(
            initial_portfolio_value=1000,
            full_data=base_comb_df,
            today_date="2022-02-01",
            start_est_date="2022-01-03",
            calc_share_returns_regexp=r"interdaily_pct_change$",
            physical_share_returns_regexp=r"interdaily_pct_change$",
            riskfree_rate_col="daily_riskless_yield",
            weight_comp_kwargs={"required_return": 0.005},
        )
    )

    assert isinstance(incremented_portfolio_value, float)
    assert round(incremented_portfolio_value, 2) == 1024.2

    assert isinstance(today_share_weights, np.ndarray)
    assert today_share_weights.shape[0] == len(base_ticker_set)
    assert round(today_share_weights[0], 2) == 0.15

    assert isinstance(today_riskfree_weight, float)
    assert round(today_riskfree_weight, 2) == 0.64


def test_multiple_daily_reinvesting_strategy(base_comb_df, base_ticker_set):
    """Test for the multiple_daily_reinvesting_strategy function"""
    (
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
    ) = multiple_daily_reinvesting_strategy(
        initial_portfolio_value=1000,
        full_data=base_comb_df,
        start_est_date="2022-01-03",
        start_trade_date="2022-02-01",
        end_trade_date="2022-03-01",
        wieghts_recalculation_style="daily",
        calc_share_returns_regexp=r"interdaily_pct_change$",
        physical_share_returns_regexp=r"interdaily_pct_change$",
        riskfree_rate_col="daily_riskless_yield",
        combine_pure_strat_with_riskless_coef=0.1,
    )

    assert isinstance(daily_portfolio_values, list)
    assert isinstance(daily_portfolio_values[0], float)
    assert round(daily_portfolio_values[-1], 2) == 1067.95

    assert isinstance(daily_portfolio_share_weights, list)
    assert isinstance(daily_portfolio_share_weights[0], np.ndarray)
    assert isinstance(daily_portfolio_share_weights[0][0], float)
    assert round(daily_portfolio_share_weights[-1][0], 2) == 0.04
    assert daily_portfolio_share_weights[-1].shape[0] == len(base_ticker_set)

    assert isinstance(daily_portfolio_riskfree_weights, list)
    assert isinstance(daily_portfolio_riskfree_weights[0], float)
    assert round(daily_portfolio_riskfree_weights[-1], 2) == 0.1

    assert isinstance(realized_trade_dates, list)
    assert isinstance(realized_trade_dates[0], pd.Timestamp)
    assert realized_trade_dates[0] == pd.Timestamp("2022-02-01")

    assert (
        len(daily_portfolio_share_weights)
        == len(realized_trade_dates)
        == len(daily_portfolio_values)
        == len(daily_portfolio_riskfree_weights)
    )


def test_convert_multiple_reinvestment_output_to_df(
    base_multiple_daily_reinvesting_results,
):
    """Test for the convert_multiple_reinvestment_output_to_df function"""
    reinv_df = convert_multiple_reinvestment_output_to_df(
        *base_multiple_daily_reinvesting_results,
        tickers=["A", "B"],
    )

    assert isinstance(reinv_df, pd.DataFrame)

    assert reinv_df.shape[0] == 4
    assert reinv_df.shape[1] == 4

    assert reinv_df.index[0] == "2022-01-03"


@patch("matplotlib.pyplot.show")
def test_plot_multiple_reinvestment_results(
    mock_show, base_multiple_daily_reinvesting_results_df
):
    """Test for the plot_multiple_reinvestment_results function"""
    plot_multiple_reinvestment_results(
        output_df=base_multiple_daily_reinvesting_results_df,
        plot_portfolio_value=True,
        plot_weights=True,
        show_plots=True,
        show_plots_streamlit=False,
    )

    assert mock_show.call_count == 2


@patch("streamlit.pyplot")
def test_plot_multiple_reinvestment_results_streamlit(
    mock_pyplot, base_multiple_daily_reinvesting_results_df
):
    """Test for the plot_multiple_reinvestment_results function streamlit call"""
    plot_multiple_reinvestment_results(
        output_df=base_multiple_daily_reinvesting_results_df,
        plot_portfolio_value=True,
        plot_weights=True,
        show_plots=False,
        show_plots_streamlit=True,
    )

    assert mock_pyplot.call_count == 2
