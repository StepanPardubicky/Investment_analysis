# pylint: disable=C0103
"""
Second non-entry page of the app.
Contains the Portfolio Analysis of the stock and bond returns.
Specifically, it focuses on mean-variance analysis.
viz. https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
"""

import sys
import os

import streamlit as st
import pandas as pd
import numpy as np


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.data_exploration_functions import (
        get_descriptive_statistics,
        plot_weights_bar,
    )

    from functions.portfolio_functions import (
        efficient_frontier_complete,
        efficient_frontier_complete_riskless,
        compute_markowitz_weights,
        compute_sharpe_nonnegative_weights_external,
        compute_minimum_variance_portfolio,
        compute_max_sharpe_weights,
        compute_inverse_volatility_weighed_portfolio,
        compute_equal_weights_portfolio,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

##Get the values from the session state
comb_df = st.session_state["comb_df"]
return_col = st.session_state["return_col"]
start_date = st.session_state["start_date"]
today_date = st.session_state["today_date"]
end_date = st.session_state["end_date"]
tickers = sorted(st.session_state["tickers"])

start_date = pd.Timestamp(start_date)
today_date = pd.Timestamp(today_date)
end_date = pd.Timestamp(end_date)

# Cast comb_df index to pd.Timestamp
comb_df.index = pd.to_datetime(comb_df.index)

## PAGE BODY ##
st.title("Portfolio Analysis")
st.subheader("Perform mean-variance portfolio analysis on the data")

st.markdown(
    """
    This page allows you to perform mean-variance
     portfolio analysis on the stock and bond returns.
    The following functionalities are available on this page:
    
    1. **Show Sample Portfolio Weights**: Display the
     sample portfolio weights for the selected date.
    2. **Calculate Efficient Frontier**: Compute and display
      the efficient frontier for the given assets.
    3. **Calculate Capital Market Line**: Compute and
      display the capital market line (CML) for the given assets.
    4. **Compute Tangent Portfolio**: Calculate
      the weights of the tangent portfolio (maximum Sharpe ratio portfolio).
    5. **Compute Minimum Variance Portfolio**:
      Calculate the weights of the minimum variance portfolio.
    6. **Compute Equal Weights Portfolio**: 
    Calculate the weights of the equal weights portfolio.
    7. **Compute Inverse Volatility Weighed Portfolio**: 
    Calculate the weights of the inverse volatility weighed portfolio.
    """
)

st.write(
    "More details and formulas can be found eg. in the following texts "
    "https://www.karlin.mff.cuni.cz/~rusy/AnalyzaInvestic/InvestmentAnalysisNotes.pdf"
    "https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf"
)

st.sidebar.header("Portfolio Analysis Options")

show_sample_portfolio_weights = st.sidebar.checkbox(
    "Show sample portfolio weights", value=True
)

st.text(f"Portfolio analysis performed for the day {today_date.date()}")
num_days = comb_df.loc[start_date:today_date].shape[0]
st.text(f"Number of days in the analysis: {num_days}")


means, var, covs, corr = get_descriptive_statistics(
    comb_df,
    print_stats=False,
    start_date=start_date,
    end_date=today_date,
    col_regexp_substring=return_col,
)

riskless_rate = comb_df["daily_riskless_yield"].loc[today_date]

if np.all(means >= 0):
    means_min_coef = 0.1
    means_max_coef = 9
elif np.all(means <= 0):
    means_min_coef = 6
    means_max_coef = 0.1
else:
    means_min_coef = 6
    means_max_coef = 9

st.markdown("#### Efficient frontier without riskless asset (Bonds):")
st.markdown(
    """
Efficient frontier is a set of optimal portfolios 
that offer the highest expected return for a defined
 level of risk (standard deviation in this case)
 or the lowest risk for a given level of expected return.
It is constructed through combination of the selected stock.
All the weights have to sum up to 1, but shorting 
(allowing negative weights) is allowed.
"""
)
efficient_results = efficient_frontier_complete(
    means,
    covs,
    var,
    plot_results=False,
    plot_results_streamlit=True,
    means_max_coef=means_max_coef,
    means_min_coef=means_min_coef,
)

st.markdown(
    "#### Efficient frontier with riskless asset (Bonds), Capital Market Line, Tangent portfolio:"
)

st.text("Following plot shows the efficient frontier with the riskless asset.")
st.markdown(
    """
    Additionally capital market line, which results from combinations of
    riskless asset and the tangent portfolio, is shown. The set of efficient
    portfolios after the introduction of the riskless asset 
    is the capital market line.
"""
)

st.text(f"Today's quoted riskless rate: {riskless_rate}")

efficient_frontier_complete_riskless(
    means,
    covs,
    var,
    riskless_rate,
    plot_individual_stocks=True,
    plot_results=False,
    mean_tangent_mult=1.5,
    plot_results_streamlit=True,
)

if show_sample_portfolio_weights:
    st.markdown("#### Sample portfolio weights:")
    st.write(
        "Display the weights of various portfolio strategies for the given time period."
    )
    st.write(
        "Choose any number of portfolio strategies from amount the strategies below:"
    )
    portfolio_options = [
        "markowitz",
        "maximum_sharpe",
        "minimum_variance",
        "maximum_sharpe_non_negative",
        "inverse_volatility",
        "equal_weights",
    ]

    portfolios = st.multiselect(
        "Portfolio Strategies", portfolio_options, default=["markowitz"]
    )

    if "markowitz" in portfolios:
        st.markdown("###### Markowitz portfolio weights:")
        weights = compute_markowitz_weights(
            means, covs, riskless_rate, required_return=means.max() * 0.5
        )
        plot_weights_bar(weights[0], tickers, show_plot_streamlit=True)

    if "maximum_sharpe" in portfolios:
        st.markdown("###### Maximum Sharpe (Tangency) portfolio weights:")
        weights = compute_max_sharpe_weights(means, covs, riskless_rate)
        plot_weights_bar(weights, tickers, show_plot_streamlit=True)

    if "minimum_variance" in portfolios:
        st.markdown("###### Minimum Variance portfolio weights:")
        weights = compute_minimum_variance_portfolio(means, covs)
        plot_weights_bar(weights, tickers, show_plot_streamlit=True)

    if "maximum_sharpe_non_negative" in portfolios:
        st.markdown("###### Maximum Sharpe Non-Negative portfolio weights:")
        weights = compute_sharpe_nonnegative_weights_external(
            means, covs, riskless_rate
        )
        plot_weights_bar(weights, tickers, show_plot_streamlit=True)

    if "inverse_volatility" in portfolios:
        st.markdown("###### Inverse Volatility portfolio weights:")
        weights = compute_inverse_volatility_weighed_portfolio(var)
        plot_weights_bar(weights, tickers, show_plot_streamlit=True)

    if "equal_weights" in portfolios:
        st.markdown("###### Equal Weights portfolio weights:")
        weights = compute_equal_weights_portfolio(means)
        plot_weights_bar(weights[0], tickers, show_plot_streamlit=True)
