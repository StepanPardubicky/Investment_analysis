"""
This file contains functions for data manipulation.
"""

import re
import sys
import os

import yfinance as yf
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from config.constants_config import (
        BASE_TICKER_SET,
        BASE_START_DATE,
        BASE_END_DATE,
        # BASE_TODAY_DATE,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

### BASIC DATA MANIPULATION ###


def filter_by_date(
    df: pd.DataFrame, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Filter dataframe by date range.
    Assume df indexed by date.
    """
    # start_date = pd.Timestamp(start_date)
    # end_date = pd.Timestamp(end_date)

    if start_date is not None:
        df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]

    return df


def select_relevant_cols(
    df: pd.DataFrame, col_substring: str = None, col_regexp_substring: str = None
) -> pd.DataFrame:
    """
    Select relevant columns from dataframe.
    """
    if col_substring is not None:
        rel_cols = [col for col in df.columns if col_substring in col]
        df = df[rel_cols]
    if col_regexp_substring is not None:
        rel_cols = [col for col in df.columns if re.search(col_regexp_substring, col)]
        df = df[rel_cols]

    return df


### STOCKS DATA ###


def load_raw_stocks_data(
    tickers: list[str] = None,
    start_date: str = BASE_START_DATE,
    end_date: str = BASE_END_DATE,
    interval: str = "1d",
):
    """
    Fetches raw stock price data for specified tickers and date range from Yahoo Finance.

    Args:
        tickers (list[str]): List of company tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str, optional): Data interval (e.g., '1d', '1h'). Defaults to "1d".

    Returns:
        DataFrame: Stock price data.
    """
    if tickers is None:
        tickers = BASE_TICKER_SET
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    return data


def _raw_stocks_to_open_close(raw_data):
    """
    Transform raw fetched stock data to open and close prices.
    """
    raw_data.index = pd.to_datetime(raw_data.index)
    data_open = raw_data["Open"].copy()
    data_open.rename(
        columns={f"{ticker}": f"{ticker}_open" for ticker in data_open.columns},
        inplace=True,
    )
    data_close = raw_data["Close"].copy()
    data_close.rename(
        columns={f"{ticker}": f"{ticker}_close" for ticker in data_close.columns},
        inplace=True,
    )

    joined_df = pd.concat([data_open, data_close], axis=1)

    return joined_df


def _create_interdaily_pct_change(
    joined_df: pd.DataFrame, tickers: list[str] = None
) -> pd.DataFrame:
    """
    Create interdaily percentage change (Close price/Open price) for each stock in the dataset
    - Expects dataframe containing open and close prices for each stock
    """
    if tickers is None:
        tickers = [col.split("_")[0] for col in joined_df.columns if "open" in col]

    for ticker in tickers:
        joined_df[f"{ticker}_interdaily_pct_change"] = (
            joined_df[f"{ticker}_close"] / joined_df[f"{ticker}_open"] - 1
        )

    return joined_df


def _create_daily_pct_change(
    joined_df: pd.DataFrame, col_type: str = "open", tickers: list[str] = None
) -> pd.DataFrame:
    """
    Create daily percentage change for each stock in the dataset
    - Expects dataframe containing open and close prices for each stock
    - col_type: str - "open" or "close"
    """
    if tickers is None:
        tickers = [col.split("_")[0] for col in joined_df.columns if col_type in col]

    days_diff = joined_df.index.to_series().diff().dt.days
    joined_df["days_diff"] = days_diff

    for ticker in tickers:
        joined_df[f"{ticker}_daily_{col_type}_pct_change_raw"] = joined_df[
            f"{ticker}_{col_type}"
        ].pct_change()
        joined_df[f"{ticker}_daily_{col_type}_pct_change_norm"] = (
            joined_df[f"{ticker}_daily_{col_type}_pct_change_raw"] / days_diff
        )

    return joined_df


def _log_return_col(
    joined_df: pd.DataFrame,
    col_suffix: str = "interdaily_pct_change",
    tickers: list[str] = None,
) -> pd.DataFrame:
    """
    Create log-return column from predefined source col for each stock in the dataset
    """

    if tickers is None:
        tickers = [col.split("_")[0] for col in joined_df.columns if "open" in col]

    for ticker in tickers:
        joined_df[f"{ticker}_{col_suffix}_log"] = np.log(
            1 + joined_df[f"{ticker}_{col_suffix}"]
        )

    return joined_df


def transform_raw_stocks_data(
    raw_data, tickers: list[str] = None, cols_to_log: list[str] = None
) -> pd.DataFrame:
    """
    Apply all required transformations to raw stock data
    """
    if cols_to_log is None:
        cols_to_log = [
            "interdaily_pct_change",
            "daily_open_pct_change_raw",
            "daily_close_pct_change_raw",
            "daily_open_pct_change_norm",
            "daily_close_pct_change_norm",
        ]

    # Convert raw fetched to open/close prices
    joined_df = _raw_stocks_to_open_close(raw_data)

    # Use all tickers if not specified
    if tickers is None:
        tickers = [col.split("_")[0] for col in joined_df.columns if "open" in col]

    # Add interdaily pct change
    joined_df = _create_interdaily_pct_change(joined_df, tickers)

    # Add daily open pct change
    joined_df = _create_daily_pct_change(joined_df, "open", tickers)

    # Add daily close pct change
    joined_df = _create_daily_pct_change(joined_df, "close", tickers)

    # Logarithmize columns
    for col_type in cols_to_log:
        joined_df = _log_return_col(joined_df, col_type, tickers)

    return joined_df


### BONDS DATA ###


def load_raw_bond_yields_data(path="USYields2024.csv"):
    """
    Load raw bond yields data
    """
    bond_yields = pd.read_csv(path)
    bond_yields["Date"] = pd.to_datetime(bond_yields["Date"], format="%d/%m/%Y")

    bond_yields.set_index("Date", inplace=True)

    return bond_yields


def convert_anualized_rate_to_daily(
    yields_df: pd.DataFrame,
    yields_col: str = "1 Mo",
    num_days_year: int = 252,
    comp_type: str = "mult",
) -> pd.DataFrame:
    """
    Convert anualized yield rate to daily rate
    """

    # Convert percentage to multiplier
    yields_mult = yields_df[yields_col] / 100

    if comp_type == "mult":
        yields_df["daily_riskless_yield"] = (1 + yields_mult) ** (1 / num_days_year) - 1
    elif comp_type == "add":
        yields_df["daily_riskless_yield"] = yields_mult / num_days_year
    else:
        raise ValueError("Invalid type (Choose 'mult' or 'add')")

    return yields_df


def join_stocks_bonds(
    stocks_df: pd.DataFrame, bonds_df: pd.DataFrame, fill_bond_na=True
) -> pd.DataFrame:
    """
    Join stocks and bonds dataframes
    """

    # Join dataframes on index (Date)
    joined_df = stocks_df.join(bonds_df, how="left")

    # Fill missing values in "daily_riskless_yield" column by mean value
    if fill_bond_na:
        joined_df["daily_riskless_yield"] = joined_df["daily_riskless_yield"].fillna(
            joined_df["daily_riskless_yield"].mean()
        )

    return joined_df


def load_and_transform_raw_data_full(
    tickers: list[str],
    start_date: str,
    end_date: str,
    # interval: str = "1d",
    # bond_yields_path: str = "USYields2024.csv",
    # transform_raw_stocks_kwargs: dict = None,
    # convert_anualized_rate_kwargs: dict = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and transform raw stock and bond data
    """
    # Handle kwargs
    interval = kwargs.get("interval", "1d")
    bond_yields_path = kwargs.get("bond_yields_path", "USYields2024.csv")
    transform_raw_stocks_kwargs = kwargs.get("transform_raw_stocks_kwargs", None)
    convert_anualized_rate_kwargs = kwargs.get("convert_anualized_rate_kwargs", None)
    num_days_year = kwargs.get("num_days_year", 252)

    # Handle default values
    if transform_raw_stocks_kwargs is None:
        transform_raw_stocks_kwargs = {}
    if convert_anualized_rate_kwargs is None:
        convert_anualized_rate_kwargs = {}

    # Load stock data
    raw_data = load_raw_stocks_data(tickers, start_date, end_date, interval=interval)
    alt_df = transform_raw_stocks_data(raw_data, **transform_raw_stocks_kwargs)
    # Load bond data
    bond_yields = load_raw_bond_yields_data(
        path=bond_yields_path, **convert_anualized_rate_kwargs
    )
    yields_df = convert_anualized_rate_to_daily(
        bond_yields, num_days_year=num_days_year
    )
    # Join stock and bond data
    comb_df = join_stocks_bonds(alt_df, yields_df)
    return comb_df
