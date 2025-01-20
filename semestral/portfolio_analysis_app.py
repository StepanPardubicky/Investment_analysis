"""
Streamlit app for the semestral project.
Documentation for streamlit: https://docs.streamlit.io/
Parts of the code were generated using GPT
"""

import sys
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Workaround wierd python relative import behavior and pylint warnings
try:
    from config.constants_config import (
        BASE_TICKER_SET,
        BASE_START_DATE,
        BASE_TODAY_DATE,
        BASE_END_DATE,
    )

    from functions.data_manipulation_functions import (
        load_and_transform_raw_data_full,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


plt.rcParams["figure.figsize"] = [8, 6]

# Handle default values
BASE_TICKER_SET = sorted(BASE_TICKER_SET)
BASE_START_DATE = pd.to_datetime(BASE_START_DATE)
BASE_TODAY_DATE = pd.to_datetime(BASE_TODAY_DATE)
BASE_END_DATE = pd.to_datetime(BASE_END_DATE)

# Title of the app
st.title("Portfolio Management App")

### SIDEBAR ###

# Sidebar with user inputs
st.sidebar.header("User Input Widgets")

use_defaults = st.sidebar.checkbox(
    "Use Default Values",
    value=st.session_state.get("use_defaults", True),
)

if use_defaults:
    st.sidebar.write(
        "Default values will be used. This serves as a fast start to get oreinted with the app."
    )
    start_date = BASE_START_DATE
    st.session_state["start_date"] = start_date
    end_date = BASE_END_DATE
    st.session_state["end_date"] = end_date
    today_date = BASE_TODAY_DATE
    st.session_state["today_date"] = today_date
    tickers = BASE_TICKER_SET
    st.session_state["tickers"] = tickers
    USE_DEFAULT_TICKERS = True
    # Following variables are not really constants
    return_col = r"interdaily_pct_change$"  # pylint: disable=C0103
    physical_return_col = return_col  # pylint: disable=C0103
    st.session_state["return_col"] = return_col
    st.session_state["physical_return_col"] = return_col

else:
    st.sidebar.write("Custom values will be used.")
    # Date inputs

    start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.get("start_date", BASE_START_DATE),
        max_value=BASE_TODAY_DATE,
        min_value=BASE_START_DATE,
        format="YYYY-MM-DD",
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state.get("end_date", BASE_END_DATE),
        max_value=BASE_END_DATE,
        min_value=BASE_TODAY_DATE,
        format="YYYY-MM-DD",
    )

    today_date = st.sidebar.date_input(
        "Today's Date",
        value=st.session_state.get("today_date", BASE_TODAY_DATE),
        max_value=end_date,
        min_value=start_date,
        format="YYYY-MM-DD",
    )

    USE_DEFAULT_TICKERS = st.sidebar.checkbox(
        "Use Default Tickers",
        value=st.session_state.get("use_default_tickers", True),
    )

    if not USE_DEFAULT_TICKERS:
        tickers_raw = st.sidebar.text_area(
            "Enter a list of valid tickers (comma-separated):",
            (",").join(st.session_state.get("tickers", BASE_TICKER_SET)),
        )
        tickers = tickers_raw.split(",")
    else:
        tickers = BASE_TICKER_SET

    st.sidebar.write("Selected Tickers:", tickers)

    return_col_options = [
        r"interdaily_pct_change$",
        r"daily_open_pct_change_raw$",
        r"daily_close_pct_change_raw$",
        r"daily_open_pct_change_norm$",
        r"daily_close_pct_change_norm$",
        r"interdaily_pct_change_log$",
        r"daily_open_pct_change_raw_log$",
        r"daily_close_pct_change_raw_log$",
        r"daily_open_pct_change_norm_log$",
        r"daily_close_pct_change_norm_log$",
    ]
    return_col = st.sidebar.selectbox(
        "Return Collumn",
        return_col_options,
        index=return_col_options.index(
            st.session_state.get("return_col", r"interdaily_pct_change$")
        ),
    )

    physical_return_col = st.sidebar.selectbox(
        "Physical Return Column",
        [
            r"interdaily_pct_change$",
            r"daily_open_pct_change_raw$",
            r"daily_close_pct_change_raw$",
            r"daily_open_pct_change_norm$",
            r"daily_close_pct_change_norm$",
        ],
        index=return_col_options.index(
            st.session_state.get("physical_return_col", r"interdaily_pct_change$")
        ),
    )

### Load the data###

if use_defaults:
    comb_df = pd.read_csv("semestral\\comb_df.csv", index_col=0)

else:
    comb_df = load_and_transform_raw_data_full(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        bond_yields_path="semestral\\USYields2024.csv",
    )


st.session_state["use_defaults"] = use_defaults
st.session_state["start_date"] = start_date
st.session_state["end_date"] = end_date
st.session_state["today_date"] = today_date
st.session_state["tickers"] = tickers
st.session_state["comb_df"] = comb_df
st.session_state["return_col"] = return_col
st.session_state["physical_return_col"] = physical_return_col
st.session_state["use_default_tickers"] = USE_DEFAULT_TICKERS

## MAIN PAGE  - Exploratory Analysis###

st.subheader("Author: Štěpán Pardubický")
st.subheader(
    "This application serves as semestral project for the course *BI-PYT Programování v Pythonu*"
)

st.markdown(
    "## Welcome to the Portfolio Management App! \n"
    "### This is the entry page for the app. \n"
    "Current page serves mainly as an starting point for the app. "
    "(To change the pages navigate to the sidebar) "
    "Additionally, in the sidebar you can find the user input widgets "
    "to change the default values. "
    "For first time usage, it is suggested to keep the default values. "
    "Afterwards feel free to experiment with the widgets (setting custom stock ticker). "
)

st.subheader("Pages contents description:")

st.markdown(
    """
    1. **Exploratory Data Analysis**: Perform exploratory data analysis (EDA)
      on the stock and bond returns. This includes visualizing stock value development, 
      calculating descriptive statistics, and identifying missing and zero values.
    2. **Portfolio Analysis**: Perform mean-variance portfolio analysis
      on the stock and bond returns. This includes calculating the efficient frontier,
        capital market line, and tangent portfolio, 
        as well as visualizing sample calculated portfolio weights.
    3. **Portfolio Value Evolution**: Visualize the evolution of the
      portfolio constructed via chosen strategy over time.
        This includes calculating daily portfolio values, portfolio weights,
          and comparing different investment strategies.
    4. **Anomaly Detection**: Perform anomaly detection on stock and
      bond returns using autoencoder models. This includes detecting anomalies
        in individual stocks as well as jointly modeling multiple
          stocks to capture relationships between their returns.
    """
)


with st.expander("User Input Widget Description", expanded=True):
    st.markdown(
        """
        - **Use Default Values**: Checkbox to use default
          values for the user input widgets.
        - **Start Date**: Date input widget for the start date of the data.
        - **End Date**: Date input widget for the end date of the data.
        - **Today's Date**: Date input widget for the date of the analysis.
        - **Use Default Tickers**: Checkbox to use default tickers for the analysis.
        - **Enter a list of valid tickers (comma-separated)**: Text area to
          enter a list of valid tick
        - **Return Column**: Dropdown menu to select the return column for
          the analysis.
        - **Physical Return Column**: Dropdown menu to select the physical
          return column for the analysis.
    """
    )


st.markdown("#### Selected widget values:")
st.write(f"Start date: {start_date}")
st.write(f"End date: {end_date}")
st.write(f"Today's date: {today_date}")
st.write(f"Selected tickers: {tickers}")
st.write(f"Return column selected: {return_col}")
st.write(f"Physical return column selected: {physical_return_col}")
