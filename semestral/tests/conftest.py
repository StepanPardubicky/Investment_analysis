"""
Set base test fixtures
"""

# pylint: disable=W0621

import sys
import os

import pandas as pd
import pytest
import numpy as np


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from config.constants_config import (
        BASE_TICKER_SET as config_base_ticker_set,
        BASE_START_DATE as config_base_start_date,
        BASE_TODAY_DATE as config_base_today_date,
        BASE_END_DATE as config_base_end_date,
    )

    from functions.data_manipulation_functions import (
        load_raw_stocks_data,
        transform_raw_stocks_data,
        load_raw_bond_yields_data,
        convert_anualized_rate_to_daily,
        filter_by_date,
    )

    from functions.data_exploration_functions import get_descriptive_statistics

    from functions.anomaly_detection_functions import (
        _build_autoencoder_model,
        get_joint_autoencode_anomaly,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


@pytest.fixture
def base_ticker_set():
    """Base set of tickers fixture"""
    return sorted(config_base_ticker_set)


@pytest.fixture
def base_start_date():
    """Base start date fixture"""
    return config_base_start_date


@pytest.fixture
def base_today_date():
    """Base today date fixture"""
    return config_base_today_date


@pytest.fixture
def base_end_date():
    """Base end date fixture"""
    return config_base_end_date


@pytest.fixture
def base_raw_data():
    """Base raw stock data fixture"""
    return load_raw_stocks_data(
        tickers=config_base_ticker_set,
        start_date=config_base_start_date,
        end_date=config_base_end_date,
        interval="1d",
    )


@pytest.fixture
def base_transformed_data(base_raw_data):
    """Base transformed data fixture"""
    return transform_raw_stocks_data(base_raw_data)


@pytest.fixture
def base_raw_bond_yields():
    """Base raw bond yields data fixture"""
    return load_raw_bond_yields_data("semestral/USYields2024.csv")


@pytest.fixture
def base_riskless_rate_bond_yields(base_raw_bond_yields):
    """Base riskless rate bond yields data fixture"""
    return convert_anualized_rate_to_daily(base_raw_bond_yields)


# @pytest.fixture
# def base_comb_df(base_ticker_set, base_start_date, base_end_date):
#     """Base combined dataframe fixture"""
#     return load_and_transform_raw_data_full(
#         tickers=base_ticker_set,
#         start_date=base_start_date,
#         end_date=base_end_date,
#         bond_yields_path="semestral/USYields2024.csv",
#     )


@pytest.fixture
def base_comb_df():
    """Base combined dataframe fixture"""
    df = pd.read_csv("semestral/comb_df.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


@pytest.fixture
def base_dummy_small_df():
    """Base dummy small dataframe fixture"""
    return pd.DataFrame(
        {
            "A": [0, 2, 7],
            "B": [4, 5, 0],
            "C": [7, 8, 9],
        }
    )


@pytest.fixture
def base_comb_descriptive_stats(base_comb_df, base_start_date, base_today_date):
    """Base comb df means fixture"""
    descriptive_stats = get_descriptive_statistics(
        df=base_comb_df,
        start_date=base_start_date,
        end_date=base_today_date,
        print_result_streamlit=False,
        col_regexp_substring="interdaily_pct_change$",
    )
    return descriptive_stats


@pytest.fixture
def base_comb_means(base_comb_descriptive_stats):
    """Base comb df means fixture"""
    return base_comb_descriptive_stats[0]


@pytest.fixture
def base_comb_var(base_comb_descriptive_stats):
    """Base comb df variances fixture"""
    return base_comb_descriptive_stats[1]


@pytest.fixture
def base_comb_cov(base_comb_descriptive_stats):
    """Base comb df covariances fixture"""
    return base_comb_descriptive_stats[2]


@pytest.fixture
def base_comb_corr(base_comb_descriptive_stats):
    """Base comb df correlations fixture"""
    return base_comb_descriptive_stats[3]


@pytest.fixture
def simple_weights():
    """Simple weights fixture"""
    return np.array([0.5, 0.5])


@pytest.fixture
def simple_means():
    """Simple means fixture"""
    return np.array([0.1, 0.2])


@pytest.fixture
def simple_cov_matrix():
    """Simple covariance matrix fixture"""
    return np.array([[0.1, 0.05], [0.05, 0.1]])


@pytest.fixture
def simple_var():
    """Simple variance fixture"""
    return np.array([0.1, 0.1])


@pytest.fixture
def simple_share_returns():
    """Simple daily returns fixture"""
    return np.array([0.11, 0.18])


@pytest.fixture
def base_multiple_daily_reinvesting_results():
    """Base multiple daily reinvesting results fixture"""
    daily_portfolio_values = [1000, 1100, 1200, 1300]
    daily_portfolio_share_weights = [
        np.array([0.45, 0.45]),
        np.array([0.4, 0.5]),
        np.array([0.35, 0.55]),
        np.array([0.3, 0.6]),
    ]
    daily_portfolio_riskfree_weights = [0.1, 0.1, 0.1, 0.1]
    realized_trade_dates = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]

    return (
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
    )


@pytest.fixture
def base_multiple_daily_reinvesting_results_df(base_multiple_daily_reinvesting_results):
    """Base multiple daily reinvesting results dataframe fixture"""
    return pd.DataFrame(
        {
            "portfolio_value": base_multiple_daily_reinvesting_results[0],
            "share_weights": base_multiple_daily_reinvesting_results[1],
            "riskless_weights": base_multiple_daily_reinvesting_results[2],
            "trade_date": base_multiple_daily_reinvesting_results[3],
        }
    )


@pytest.fixture
def anomaly_sub_df(base_comb_df):
    """Base anomaly sub dataframe fixture"""
    sub_df = base_comb_df[["COST_interdaily_pct_change", "ANSS_interdaily_pct_change"]]
    sub_df = filter_by_date(sub_df, "2022-01-03", "2022-04-01")
    return sub_df


@pytest.fixture
def anomaly_model(anomaly_sub_df):
    """Base anomaly model fixture"""
    return _build_autoencoder_model(
        scaled_returns=anomaly_sub_df,  # Are not scaled for the test, shouldnt matter
        model_input_dim=2,
        encoding_dim=1,
        hidden_layer_sizes=(8, 2),
        activation="relu",
        num_epochs=1,
        batch_size=6,
    )


@pytest.fixture
def anomaly_test_get_joint_autoencode_results(anomaly_sub_df):
    """Base anomaly test get joint autoencode results fixture"""
    returns_data, anomaly_df = get_joint_autoencode_anomaly(
        df=anomaly_sub_df,
        col_regexp_substring=r"interdaily_pct_change$",
        encoding_dim=1,
        hidden_layer_sizes=(8, 2),
        activation="relu",
        num_epochs=1,
        batch_size=6,
    )

    return returns_data, anomaly_df
