"""
Test for data import and manipulation functions
"""

# import pytest

import sys
import os

from pandas import isna


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.data_manipulation_functions import (
        load_raw_stocks_data,
        transform_raw_stocks_data,
        _raw_stocks_to_open_close,
        load_raw_bond_yields_data,
        convert_anualized_rate_to_daily,
        join_stocks_bonds,
        load_and_transform_raw_data_full,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def test_tickers_is_list_strings(base_ticker_set):
    """Test that the tickers are a list of strings"""
    assert isinstance(base_ticker_set, list)
    assert all(isinstance(ticker, str) for ticker in base_ticker_set)


def test_load_raw_data(base_ticker_set, base_start_date, base_end_date):
    """Test loading raw data from Yahoo Finance"""
    raw_data = load_raw_stocks_data(
        tickers=base_ticker_set, start_date=base_start_date, end_date=base_end_date
    )

    assert not raw_data.empty
    assert raw_data.shape[0] > 0


def test_raw_stocks_to_open_close(base_raw_data):
    """Test transforming the raw dataset into open and clos prices"""
    open_close_data = _raw_stocks_to_open_close(base_raw_data)

    assert not open_close_data.empty
    assert open_close_data.shape[0] > 0
    assert len([column for column in open_close_data.columns if "open" in column]) > 0
    assert len([column for column in open_close_data.columns if "close" in column]) > 0


def test_transform_raw_data(base_raw_data):
    """Test transforming the raw dataset"""
    transformed_data = transform_raw_stocks_data(base_raw_data)

    assert not transformed_data.empty
    assert transformed_data.shape[0] > 0


def test_load_raw_bond_yields_data():
    """Test loading raw bond yields data"""
    load_raw_bonds_df = load_raw_bond_yields_data("semestral/USYields2024.csv")

    assert not load_raw_bonds_df.empty
    assert "1 Mo" in load_raw_bonds_df.columns
    assert load_raw_bonds_df.shape[0] > 0
    assert load_raw_bonds_df.shape[1] > 0


def test_convert_anualized_rate_to_daily(base_raw_bond_yields):
    """Test converting anualized rate to daily"""
    converted_df = convert_anualized_rate_to_daily(base_raw_bond_yields)

    assert not converted_df.empty
    assert converted_df.shape[0] > 0
    assert converted_df.shape[1] > 0
    assert "daily_riskless_yield" in converted_df.columns
    assert converted_df["daily_riskless_yield"].iloc[0] > 0


def test_join_stocks_bonds(base_transformed_data, base_riskless_rate_bond_yields):
    """Test joining stocks and bonds dataframes"""
    joined_df = join_stocks_bonds(base_transformed_data, base_riskless_rate_bond_yields)

    assert not joined_df.empty
    assert joined_df.shape[0] > 0
    assert joined_df.shape[1] > 0
    assert "COST_open" in joined_df.columns
    assert "daily_riskless_yield" in joined_df.columns


def test_load_and_transform_raw_data_full(
    base_ticker_set, base_start_date, base_end_date
):
    """Test loading and transforming raw stock and bond data"""
    full_loaded_df = load_and_transform_raw_data_full(
        tickers=base_ticker_set,
        start_date=base_start_date,
        end_date=base_end_date,
        bond_yields_path="semestral/USYields2024.csv",
    )

    assert not full_loaded_df.empty

    assert full_loaded_df.shape[0] > 0
    assert full_loaded_df.shape[1] > 0

    assert "COST_open" in full_loaded_df.columns
    assert "daily_riskless_yield" in full_loaded_df.columns
    assert "COST_interdaily_pct_change" in full_loaded_df.columns
    assert "COST_daily_close_pct_change_norm" in full_loaded_df.columns
    assert "COST_daily_close_pct_change_raw_log" in full_loaded_df.columns

    assert full_loaded_df["daily_riskless_yield"].iloc[0] > 0
    assert full_loaded_df["COST_open"].iloc[0] > 0
    assert full_loaded_df["COST_interdaily_pct_change"].iloc[0] > 0

    assert isna(full_loaded_df["COST_daily_close_pct_change_norm"].iloc[0])
    assert not isna(full_loaded_df["COST_daily_close_pct_change_norm"].iloc[1])
    assert (
        full_loaded_df["COST_daily_close_pct_change_norm"].iloc[1] > -1
        and full_loaded_df["COST_daily_close_pct_change_norm"].iloc[1] < 1
    )

    assert isna(full_loaded_df["COST_daily_close_pct_change_raw_log"].iloc[0])
    assert not isna(full_loaded_df["COST_daily_close_pct_change_raw_log"].iloc[1])
    assert (
        full_loaded_df["COST_daily_close_pct_change_raw_log"].iloc[1] > -1
        and full_loaded_df["COST_daily_close_pct_change_raw_log"].iloc[1] < 1
    )
