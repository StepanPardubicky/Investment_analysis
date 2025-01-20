# pylint: disable=no-member
"""
This file contains functions for data exploration.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import streamlit as st


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Workaround wierd python relative import behavior and pylint warnings
try:
    from functions.data_manipulation_functions import (
        filter_by_date,
        select_relevant_cols,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def get_missing_and_zero_values(
    df,
    start_date=None,
    end_date=None,
    # get_missing=True,
    # get_zero=True,
    # print_results=True,
    # print_result_streamlit=False,
    # col_substring: str = None,
    # col_regexp_substring: str = None,
    **kwargs,
) -> None:
    """
    Get missing and zero values from a dataframe
    """

    # Handle kwargs
    get_missing = kwargs.get("get_missing", True)
    get_zero = kwargs.get("get_zero", True)
    print_results = kwargs.get("print_results", True)
    print_result_streamlit = kwargs.get("print_result_streamlit", False)
    col_substring = kwargs.get("col_substring", None)
    col_regexp_substring = kwargs.get("col_regexp_substring", None)

    # Only select the relevant columns
    df = filter_by_date(df, start_date, end_date)

    # Filter for relevant dates
    df = select_relevant_cols(df, col_substring, col_regexp_substring)

    missing_values = df.isnull().sum()
    zero_values = (df == 0).sum()
    if print_results:
        if get_missing:
            print("Following missing values were found:")
            print(f"Missing values: {missing_values[missing_values > 0]}")
        if get_zero:
            print("Following zero values were found:")
            print(f"Zero values: {zero_values[zero_values > 0]}")

    if print_result_streamlit:
        if get_missing:
            st.write("Following missing values were found:")
            st.write(f"Missing values: {missing_values[missing_values > 0]}")
        if get_zero:
            st.write("Following zero values were found:")
            st.write(f"Zero values: {zero_values[zero_values > 0]}")


def plot_df(
    df: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    **kwargs,
) -> None:
    """
    Plot a dataframe
    """

    # Handle kwargs
    title = kwargs.get("title", None)
    col_substring = kwargs.get("col_substring", None)
    col_regexp_substring = kwargs.get("col_regexp_substring", None)
    show_plot = kwargs.get("show_plot", True)
    show_plot_streamlit = kwargs.get("show_plot_streamlit", False)

    # Only select the relevant columns
    df = filter_by_date(df, start_date, end_date)

    # Filter for relevant dates
    df = select_relevant_cols(df, col_substring, col_regexp_substring)

    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    df.plot(ax=ax)

    if show_plot_streamlit:
        st.pyplot(fig)

    if show_plot:
        plt.show()


def get_descriptive_statistics(
    df: pd.DataFrame,
    # print_stats: bool = False,
    # start_date: str = None,
    # end_date: str = None,
    # col_substring: str = None,
    # col_regexp_substring: str = None,
    **kwargs,
):
    """
    Get basic descriptive statistics for the dataframe
    """

    # Handle kwargs
    print_stats = kwargs.get("print_stats", False)
    start_date = kwargs.get("start_date", None)
    end_date = kwargs.get("end_date", None)
    col_substring = kwargs.get("col_substring", None)
    col_regexp_substring = kwargs.get("col_regexp_substring", None)

    # Only select the relevant columns
    df = filter_by_date(df, start_date, end_date)
    # Filter for relevant dates
    df = select_relevant_cols(df, col_substring, col_regexp_substring)

    means = df.mean()
    var = df.var()
    covs = df.cov()
    corr = df.corr()

    cond_numb = np.linalg.cond(
        covs
    )  ##Matrix condition number is sufficiently small to safely construct inverse matrix
    if cond_numb > 40:  # 40 has been chosen relatively arbitrarily
        print(
            f"Warning: The condition number of the covariance matrix ({cond_numb}) is high."
        )
        print(
            "This may lead to unstable inverse matrix computation and unreliable results."
        )

    if print_stats:
        print("Means: ")
        print(means)
        print("Variances: ")
        print(var)
        print(f"Conditional number of the covariance matrix: {cond_numb}")
        print("Df describe results: ")
        print(df.describe())

    return means, var, covs, corr


def barplot_means(
    means: pd.Series,
    title: str = "Means of the different columns",
    show_plot: bool = True,
    show_plot_streamlit: bool = False,
    rotatex: bool = False,
):
    """
    Plot the means of the different columns
    """
    fig, ax = plt.subplots()
    # Following pylint error is suppressed as it is faulty
    means.plot(kind="bar", color=plt.cm.tab20.colors, ax=ax)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    if rotatex:
        plt.xticks(rotation=45)
    ax.grid(linewidth=0.5)

    if show_plot_streamlit:
        st.pyplot(fig)

    if show_plot:
        plt.show()


def plot_covariances_matrix(
    var: pd.Series,
    corr: pd.DataFrame,
    # plot_variance_bar: bool = True,
    # plot_heatmap: bool = True,
    # show_plot: bool = True,
    # show_plot_streamlit: bool = False,
    **kwargs,
):
    """
    Plot the covariance
    - If plot_variance_bar is True, plot the variance of the different columns
    - If plot_heatmap is True, plot the heatmap of the correlation matrix
    """

    # Handle kwargs
    plot_variance_bar = kwargs.get("plot_variance_bar", True)
    plot_heatmap = kwargs.get("plot_heatmap", True)
    show_plot = kwargs.get("show_plot", True)
    show_plot_streamlit = kwargs.get("show_plot_streamlit", False)

    if plot_variance_bar:
        fig, ax = plt.subplots()
        # pylint: disable=no-member
        var.plot(kind="bar", color=plt.cm.tab20.colors, ax=ax)
        ax.set_title("Variance of the different columns")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(linewidth=0.5)

        if show_plot_streamlit:
            st.pyplot(fig)

        if show_plot:
            plt.show()

    if plot_heatmap:
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Heatmap of Correlation Matrix")

        if show_plot_streamlit:
            st.pyplot(fig)

        if show_plot:
            plt.show()


def plot_autocorrelation(
    df: pd.DataFrame,
    num_lags: int = 20,
    start_date: str = None,
    end_date: str = None,
    # col_substring: str = None,
    # col_regexp_substring: str = None,
    # show_plot: bool = True,
    # show_plot_streamlit: bool = False,
    **kwargs,
):
    """
    Plot the autocorrelation function of the dataframe
    """

    # Handle kwargs
    col_substring = kwargs.get("col_substring", None)
    col_regexp_substring = kwargs.get("col_regexp_substring", None)
    show_plot = kwargs.get("show_plot", True)
    show_plot_streamlit = kwargs.get("show_plot_streamlit", False)

    # Only select the relevant columns
    df = filter_by_date(df, start_date, end_date)
    # Filter for relevant dates
    df = select_relevant_cols(df, col_substring, col_regexp_substring)

    for col in df.columns:
        fig, ax = plt.subplots()
        plot_acf(df[col], lags=num_lags, ax=ax)
        ax.set_title(f"Autocorrelation function of {col}")

        if show_plot_streamlit:
            st.pyplot(fig)

        if show_plot:
            plt.show()


def plot_weights_bar(
    weights: np.ndarray,
    tickers: list,
    title: str = "Portfolio Weights",
    show_plot: bool = True,
    show_plot_streamlit: bool = False,
):
    """
    Plot the portfolio weights
    """
    fig, ax = plt.subplots()
    ax.bar(tickers, weights)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(linewidth=0.5)

    if show_plot_streamlit:
        st.pyplot(fig)

    if show_plot:
        plt.show()
