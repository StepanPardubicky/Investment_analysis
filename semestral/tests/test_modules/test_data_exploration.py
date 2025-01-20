"""
Tests for data exploration functions
Some part of this code were coauthored by GPT-4o
"""

import sys
import os
from io import StringIO
from unittest.mock import patch

# import pytest


import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.data_exploration_functions import (
        get_missing_and_zero_values,
        plot_df,
        get_descriptive_statistics,
        barplot_means,
        plot_covariances_matrix,
        plot_autocorrelation,
        plot_weights_bar,
    )
    from functions.data_manipulation_functions import (
        filter_by_date,
        select_relevant_cols,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Block plots from showing up
matplotlib.use("Agg")


##Following patch/mock_stdout approach was suggested by GPT-4o
def test_get_missing_zero_values_base_print(base_comb_df):
    """Test getting missing and zero values from a dataframe"""

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        get_missing_and_zero_values(
            df=base_comb_df,
            start_date="2022-01-01",
            end_date="2022-01-31",
            print_results=True,
            print_result_streamlit=False,
        )

        output = mock_stdout.getvalue()

        assert "Following missing values were found:" in output
        assert "Following zero values were found:" in output
        assert "Missing values: " in output
        assert "Zero values: " in output


# Following test was suggested by GPT-4o
def test_get_missing_zero_values_dummy_print():
    """Test getting missing and zero values from a custom dummy dataframe"""
    # Create a test dataframe
    df = pd.DataFrame(
        {
            "A": [1, 0, None],
            "B": [0, 2, 3],
            "C": [4, 5, 6],
        }
    )

    # Mock sys.stdout to capture print output
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        get_missing_and_zero_values(
            df, print_results=True, get_missing=True, get_zero=True
        )

        # Get the captured output
        output = mock_stdout.getvalue()

        # Assertions on the output
        assert "Following missing values were found:" in output
        assert "Missing values: A    1" in output  # Expected missing value
        assert "Following zero values were found:" in output
        assert "Zero values: A    1" in output  # Expected zero values


# Following test was suggested by GPT-4o
@patch("streamlit.write")
def test_get_missing_zero_values_streamlit(mock_write):
    """Test getting missing and zero values printing to Streamlit"""
    # Create a test dataframe
    df = pd.DataFrame(
        {
            "A": [1, 0, None],
            "B": [0, 2, 3],
            "C": [4, 5, 6],
        }
    )

    # Call the function with `print_result_streamlit=True`
    get_missing_and_zero_values(
        df, print_result_streamlit=True, get_missing=True, get_zero=True
    )

    # Assertions on the Streamlit calls
    assert mock_write.call_count == 4  # Two headers and two data outputs
    mock_write.assert_any_call("Following missing values were found:")
    mock_write.assert_any_call("Following zero values were found:")


# Creation of following test was assisted by gpt-4o
@patch("matplotlib.pyplot.show")
def test_plot_df(
    mock_show,
    base_comb_df,
    base_start_date,
    base_end_date,
):
    """Basic test_plot_df test"""

    plot_df(
        df=base_comb_df,
        start_date=base_start_date,
        end_date=base_end_date,
        title="Test Plot",
        show_plot=True,
        show_plot_streamlit=False,
        col_regexp_substring="interdaily_pct_change$",
    )

    mock_show.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Test Plot"


@patch("streamlit.pyplot")
def test_plot_df_streamlit(
    mock_pyplot,
    base_comb_df,
    base_start_date,
    base_end_date,
):
    """Streamlit plot_df test"""

    plot_df(
        df=base_comb_df,
        start_date=base_start_date,
        end_date=base_end_date,
        title="Test Plot",
        show_plot=False,
        show_plot_streamlit=True,
        col_regexp_substring="interdaily_pct_change$",
    )

    mock_pyplot.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Test Plot"


def test_descriptive_statistics_base(base_comb_df, base_start_date, base_today_date):
    """Test descriptive statistics function"""

    mean, var, cov, corr = get_descriptive_statistics(
        df=base_comb_df,
        start_date=base_start_date,
        end_date=base_today_date,
        print_result_streamlit=False,
        col_regexp_substring="interdaily_pct_change$",
    )

    assert mean is not None
    assert var is not None
    assert cov is not None
    assert corr is not None

    assert isinstance(mean, pd.Series)
    assert isinstance(var, pd.Series)
    assert isinstance(cov, pd.DataFrame)
    assert isinstance(corr, pd.DataFrame)

    filt_df = filter_by_date(base_comb_df, base_start_date, base_today_date)
    filt_df = select_relevant_cols(
        filt_df, col_regexp_substring="interdaily_pct_change$"
    )

    assert mean.shape[0] == filt_df.shape[1]
    assert var.shape[0] == filt_df.shape[1]
    assert cov.shape[0] == filt_df.shape[1]
    assert cov.shape[1] == filt_df.shape[1]
    assert corr.shape[0] == filt_df.shape[1]
    assert corr.shape[1] == filt_df.shape[1]

    assert (var >= 0).all()
    assert (corr >= -1).all(axis=None)
    assert (corr <= 1).all(axis=None)


def test_descriptive_statistics_dummy(base_dummy_small_df):
    """Test descriptive statistics function values on small_dummy_df"""
    mean, var, cov, corr = get_descriptive_statistics(
        df=base_dummy_small_df,
        print_result_streamlit=False,
    )

    assert (mean == pd.Series([3, 3, 8], index=["A", "B", "C"])).all()

    assert var.loc["A"] == 13.0

    assert cov.loc["A", "B"] == -8.5

    assert corr.loc["A", "B"] == -0.8910421112136306  # Copied from the test result


@patch("matplotlib.pyplot.show")
def test_barplot_means_base(mock_show, base_comb_means):
    """Test barplot_means function"""
    barplot_means(
        means=base_comb_means,
        title="Test Barplot",
        show_plot=True,
        show_plot_streamlit=False,
    )

    mock_show.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Test Barplot"
    assert len(ax.patches) == len(base_comb_means)


@patch("streamlit.pyplot")
def test_barplot_means_streamlit(mock_pyplot, base_comb_means):
    """Test barplot_means function with Streamlit"""
    barplot_means(
        means=base_comb_means,
        title="Test Barplot",
        show_plot=False,
        show_plot_streamlit=True,
    )

    mock_pyplot.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Test Barplot"
    assert len(ax.patches) == len(base_comb_means)


@patch("matplotlib.pyplot.show")
def test_plot_covariances_matrix_variance_bar_base(
    mock_show, base_comb_var, base_comb_corr
):
    """Test plot_covariances_matrix function's variance bar plot"""
    plot_covariances_matrix(
        var=base_comb_var,
        corr=base_comb_corr,
        plot_variance_bar=True,
        plot_heatmap=False,
        show_plot=True,
        show_plot_streamlit=False,
    )

    mock_show.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Variance of the different columns"
    assert len(ax.patches) == len(base_comb_var)


@patch("matplotlib.pyplot.show")
def test_plot_covariances_matrix_heatmap_base(mock_show, base_comb_var, base_comb_corr):
    """Test plot_covariances_matrix function's heatmap plot"""
    plot_covariances_matrix(
        var=base_comb_var,
        corr=base_comb_corr,
        plot_variance_bar=False,
        plot_heatmap=True,
        show_plot=True,
        show_plot_streamlit=False,
    )

    mock_show.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Heatmap of Correlation Matrix"


@patch("streamlit.pyplot")
def test_plot_covariance_matrix_variance_bar_streamlit(
    mock_pyplot, base_comb_var, base_comb_corr
):
    """Test plot_covariances_matrix function with Streamlit (variance bar plot)"""
    plot_covariances_matrix(
        var=base_comb_var,
        corr=base_comb_corr,
        plot_variance_bar=True,
        plot_heatmap=False,
        show_plot=False,
        show_plot_streamlit=True,
    )

    mock_pyplot.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Variance of the different columns"
    assert len(ax.patches) == len(base_comb_var)


@patch("streamlit.pyplot")
def test_plot_covariance_matrix_heatmap_streemlit(
    mock_pyplot, base_comb_var, base_comb_corr
):
    """Test plot_covariances_matrix function with Streamlit (heatmap)"""
    plot_covariances_matrix(
        var=base_comb_var,
        corr=base_comb_corr,
        plot_variance_bar=False,
        plot_heatmap=True,
        show_plot=False,
        show_plot_streamlit=True,
    )

    mock_pyplot.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Heatmap of Correlation Matrix"
    plt.close()


# @patch("matplotlib.pyplot.show")
# def test_plot_autocorrelation_base(mock_show, base_comb_df):
#     """Test plot_autocorrelation function"""
#     plot_autocorrelation(
#         df=base_comb_df,
#         num_lags=20,
#         col_regexp_substring="interdaily_pct_change$",
#         show_plot=True,
#         show_plot_streamlit=False,
#     )

#     assert mock_show.call_count == 8

#     fig = plt.gcf()
#     ax = fig.gca()
#     assert "Autocorrelation" in ax.get_title()
#     assert len(ax.lines) > 0
#     plt.close()


@patch("matplotlib.pyplot.show")
def test_plot_autocorrelation_base(mock_show, base_dummy_small_df):
    """Test plot_autocorrelation function"""
    plot_autocorrelation(
        df=base_dummy_small_df,
        num_lags=1,
        show_plot=True,
        show_plot_streamlit=False,
    )

    assert mock_show.call_count == 3

    fig = plt.gcf()
    ax = fig.gca()
    assert "Autocorrelation" in ax.get_title()
    plt.close()


# @patch("streamlit.pyplot")
# def test_plot_autocorrelation_streamlit(mock_pyplot, base_comb_df):
#     """Test plot_autocorrelation function with Streamlit"""
#     plot_autocorrelation(
#         df=base_comb_df,
#         num_lags=20,
#         col_regexp_substring="interdaily_pct_change$",
#         show_plot=False,
#         show_plot_streamlit=True,
#     )

#     assert mock_pyplot.call_count ==8

#     fig = plt.gcf()
#     ax = fig.gca()
#     assert "Autocorrelation" in ax.get_title()
#     assert len(ax.lines) > 0
#     plt.close()


@patch("streamlit.pyplot")
def test_plot_autocorrelation_streamlit(mock_pyplot, base_dummy_small_df):
    """Test plot_autocorrelation function with Streamlit"""
    plot_autocorrelation(
        df=base_dummy_small_df,
        num_lags=1,
        show_plot=False,
        show_plot_streamlit=True,
    )

    assert mock_pyplot.call_count == 3

    fig = plt.gcf()
    ax = fig.gca()
    assert "Autocorrelation" in ax.get_title()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_plot_weights_bar_base(mock_show):
    """Test plot_weights_bar function"""
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    plot_weights_bar(
        weights=weights,
        tickers=tickers,
        title="Test Barplot",
        show_plot=True,
        show_plot_streamlit=False,
    )

    mock_show.assert_called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Test Barplot"
    assert len(ax.patches) == len(tickers)
