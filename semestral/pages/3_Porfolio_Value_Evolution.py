# pylint: disable=C0103
"""
Third non-entry page of the app.
Contains the Portfolio Value Evolution of the stock and bond returns.
"""
import sys
import os

import streamlit as st
import pandas as pd


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from functions.data_exploration_functions import (
        get_descriptive_statistics,
    )

    from functions.portfolio_functions import (
        compute_markowitz_weights,
        compute_sharpe_nonnegative_weights_external,
        compute_minimum_variance_portfolio,
        compute_max_sharpe_weights,
        compute_inverse_volatility_weighed_portfolio,
        compute_equal_weights_portfolio,
    )

    from functions.portfolio_eval_functions import (
        multiple_daily_reinvesting_strategy,
        convert_multiple_reinvestment_output_to_df,
        plot_multiple_reinvestment_results,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


##Get the values from the session state
comb_df = st.session_state["comb_df"]
return_col = st.session_state["return_col"]
physical_return_col = st.session_state["physical_return_col"]
start_date = st.session_state["start_date"]
today_date = st.session_state["today_date"]
end_date = st.session_state["end_date"]
tickers = sorted(st.session_state["tickers"])

start_date = pd.Timestamp(start_date)
today_date = pd.Timestamp(today_date)
end_date = pd.Timestamp(end_date)

# Cast comb_df index to pd.Timestamp
comb_df.index = pd.to_datetime(comb_df.index)

means, var, covs, corr = get_descriptive_statistics(
    comb_df,
    print_stats=False,
    start_date=start_date,
    end_date=today_date,
    col_regexp_substring=return_col,
)


st.title("Portofolio Value Evolution")
st.subheader(
    "Visualize the evolution of the portfolio constructed via chosen strategy over time"
)


st.markdown(
    """
    This page allows you to visualize the evolution
      of the portfolio constructed via chosen strategy over time.
    The following functionalities are available on this page:
    
    1. **Show Portfolio Value Over Time**: 
    Display the portfolio value development over time.
    2. **Show Portfolio Weights Over Time**: 
    Display the portfolio weights development over time.
    3. **Compare to baseline strategies**: Compare the performance to baseline portfolios.
        - Equal weights portfolio
        - Tangent (Maximum-Sharpe) Portfolio
    """
)

st.markdown(
    """
    You can select various widget values in the sidebar

    1. **Starting capital**: The initial capital to start the portfolio with.
    2. **Portfolio strategy**: The strategy to use for the portfolio construction.
        - **Additional strategy options**: Depending on the strategy,
        additional options might be available.
    3. **Portfolio recalculation strategy**: The strategy to use for
      the portfolio recalculation. I can be:
        - **Dynamic**: Recalculate the portfolio weights daily.
        - **Fixed weights**: Keep the weights fixed. 
        (This trades to keep the weights fixed)
        - **Fixed weights with rebalancing**: Keep the portfolio fixed as is.
        The individual weights will evolve with the stock prices.
    """
)

strategy_options = [
    "markowitz",
    "maximum_sharpe",
    "minimum_variance",
    "maximum_sharpe_non_negative",
    "inverse_volatility",
    "equal_weights",
]


st.sidebar.header("Portfolio Value Evolution Options")

starting_capital = st.sidebar.number_input(
    "Starting capital",
    min_value=1000,
    max_value=1000000000,
    value=1000000,
    step=1000,
)

strategy = st.sidebar.selectbox(
    "Select the portfolio strategy",
    strategy_options,
)

if strategy == "markowitz":
    required_return = st.sidebar.slider(
        "Required mean return",
        min_value=means.min(),
        max_value=means.max() * 2,
        value=means.max() * 0.6,
        step=0.00001,
        format="%.5f",
    )


if strategy in [
    "maximum_sharpe",
    "minimum_variance",
    "maximum_sharpe_non_negative",
    "inverse_volatility",
]:
    riskless_rate = st.sidebar.slider(
        "Riskless rate",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
else:
    # False pylint alram - riskless_rate isn't really a constant
    riskless_rate = 0  # pylint: disable=invalid-name

strategy_input_dict = {
    "markowitz": compute_markowitz_weights,
    "maximum_sharpe": compute_max_sharpe_weights,
    "minimum_variance": compute_minimum_variance_portfolio,
    "maximum_sharpe_non_negative": compute_sharpe_nonnegative_weights_external,
    "inverse_volatility": compute_inverse_volatility_weighed_portfolio,
    "equal_weights": compute_equal_weights_portfolio,
}

strategy_func = strategy_input_dict.get(strategy)

# Portfolio recalculation strategy

portfolio_recalc_strategy_input = st.sidebar.selectbox(
    "Portfolio recalculation strategy",
    ["Dynamic", "Fixed weights", "Fixed weights with rebalancing"],
)

if portfolio_recalc_strategy_input == "Dynamic":
    portfolio_recalc_strategy = "daily"
elif portfolio_recalc_strategy_input == "Fixed weights":
    portfolio_recalc_strategy = "fixed_weights"
elif portfolio_recalc_strategy_input == "Fixed weights with rebalancing":
    portfolio_recalc_strategy = "fixed_portfolio"
else:
    raise ValueError("Invalid portfolio recalculation strategy")

weight_comp_kwargs = {}
if strategy == "markowitz":
    weight_comp_kwargs["required_return"] = required_return

results = multiple_daily_reinvesting_strategy(
    initial_portfolio_value=starting_capital,
    full_data=comb_df,
    start_est_date=start_date,
    start_trade_date=today_date,
    end_trade_date=end_date,
    wieghts_recalculation_style=portfolio_recalc_strategy,
    calc_share_returns_regexp=return_col,
    physical_share_returns_regexp=physical_return_col,
    weights_computation_function=strategy_func,
    combine_pure_strat_with_riskless_coef=riskless_rate,
    weight_computation_kwargs=weight_comp_kwargs,
)


(
    daily_portfolio_values,
    daily_portfolio_share_weights,
    daily_portfolio_riskfree_weights,
    realized_trade_dates,
) = results


result_df = convert_multiple_reinvestment_output_to_df(
    daily_portfolio_values,
    daily_portfolio_share_weights,
    daily_portfolio_riskfree_weights,
    realized_trade_dates,
    tickers=tickers,
)

st.write(f"Ticker symbols of the companies selected for the analysis: {tickers}")

st.write(
    f"There were {len(realized_trade_dates)} trading days observed. "
    f"First trading day: {realized_trade_dates[0].date()}, "
    f"last trading day: {realized_trade_dates[-1].date()}"
)


st.markdown("###### Portfolio evolution over time")
with st.expander("Show Portfolio value and contents over time", expanded=True):

    plot_multiple_reinvestment_results(
        result_df,
        plot_portfolio_value=True,
        plot_weights=True,
        show_plots=False,
        show_plots_streamlit=True,
    )


st.write(f"The starting capital was {starting_capital}.")

st.write(
    f"After following the selected investment strategy for {len(realized_trade_dates)} "
    f"trading days, the portfolio value changed to {daily_portfolio_values[-1]}."
)

st.write(
    f"The resulting value is {(daily_portfolio_values[-1] / starting_capital) * 100:.2f}% "
    "of the original value."
)


with st.expander(
    "Show Baseline portfolio performance: Equal weights portfolio", expanded=False
):
    results_equal_weights = multiple_daily_reinvesting_strategy(
        initial_portfolio_value=starting_capital,
        full_data=comb_df,
        start_est_date=start_date,
        start_trade_date=today_date,
        end_trade_date=end_date,
        wieghts_recalculation_style="fixed_weights",
        calc_share_returns_regexp=return_col,
        physical_share_returns_regexp=physical_return_col,
        weights_computation_function=compute_equal_weights_portfolio,
        weight_computation_kwargs={"use_riskless_asset": True},
        combine_pure_strat_with_riskless_coef=0,
    )

    (
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
    ) = results_equal_weights

    result_df = convert_multiple_reinvestment_output_to_df(
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
        tickers=tickers,
    )

    plot_multiple_reinvestment_results(
        result_df,
        plot_portfolio_value=True,
        plot_weights=True,
        show_plots=False,
        show_plots_streamlit=True,
    )

    st.write(
        f"Ther resulting baseline equal weights portfolio value is: {daily_portfolio_values[-1]},"
        f"  which is {(daily_portfolio_values[-1] / starting_capital) * 100:.2f}% "
        "of the original value."
    )

with st.expander(
    "Show Baseline portfolio performance: Tangent (Maximum-Sharpe) Portfolio",
    expanded=False,
):
    results_tangent = multiple_daily_reinvesting_strategy(
        initial_portfolio_value=starting_capital,
        full_data=comb_df,
        start_est_date=start_date,
        start_trade_date=today_date,
        end_trade_date=end_date,
        wieghts_recalculation_style="fixed_portfolio",
        calc_share_returns_regexp=return_col,
        physical_share_returns_regexp=physical_return_col,
        weights_computation_function=compute_max_sharpe_weights,
        combine_pure_strat_with_riskless_coef=0,
    )

    (
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
    ) = results_tangent

    result_df = convert_multiple_reinvestment_output_to_df(
        daily_portfolio_values,
        daily_portfolio_share_weights,
        daily_portfolio_riskfree_weights,
        realized_trade_dates,
        tickers=tickers,
    )

    plot_multiple_reinvestment_results(
        result_df,
        plot_portfolio_value=True,
        plot_weights=True,
        show_plots=False,
        show_plots_streamlit=True,
    )

    st.write(
        f"Ther resulting baseline Tangent portfolio value is: {daily_portfolio_values[-1]}, "
        f"which is {(daily_portfolio_values[-1] / starting_capital) * 100:.2f}% "
        "of the original value."
    )
