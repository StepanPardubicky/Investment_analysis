# pylint: disable=C0103
"""
First non-entry page of the app.
Contains the Exploratory Data Analysis (EDA) of the stock and bond returns.
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
        plot_df,
        get_missing_and_zero_values,
        get_descriptive_statistics,
        barplot_means,
        plot_covariances_matrix,
        plot_autocorrelation,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

st.title("Exploratory Data Analysis")
st.subheader("Perform exploratory data analysis on the data")

st.markdown(
    """
    This page allows you to perform exploratory data analysis 
    on the stock and bond returns.
    The following functionalities are available on this page:

    1. **Show Used Data**: Display the used dataframe of stock and bond returns.
    2. **Visualize Stock Value Development**: Plot the stock value 
    development over time.
    3. **Calculate Descriptive Statistics**: Compute and display
      descriptive statistics such as mean, variance, covariance, and correlation.
    4. **Identify Missing and Zero Values**: Identify and display
      missing and zero values in the dataset.
    5. **Plot Means of Different Columns**: Create bar plots 
    of the means of different columns.
    6. **Plot Variance Barplotx**: Display Barplot of the variance
    7. **Correlation Heatmap**: Plot the correlation heatmap of the dataset. 
    8. **Plot Autocorrelation**: Plot the autocorrelation 
    of the dataset.
    """
)

##Get the values from the session state
comb_df = st.session_state["comb_df"]
return_col = st.session_state["return_col"]
start_date = st.session_state["start_date"]
end_date = st.session_state["end_date"]

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Cast comb_df index to pd.Timestamp
comb_df.index = pd.to_datetime(comb_df.index)


st.sidebar.header("EDA Options")

eda_start_date = st.sidebar.date_input(
    "Exploration start date",
    value=start_date,
    min_value=start_date,
    max_value=end_date,
)
eda_end_date = st.sidebar.date_input(
    "Exploration end date",
    value=end_date,
    min_value=eda_start_date,
    max_value=end_date,
)

comb_df = comb_df.loc[eda_start_date:eda_end_date]

st.markdown(
    "#### Exploratory data analysis performed for the following date range: "
    f"{eda_start_date} - {eda_end_date}"
)

st.text(f"Number of days in the analysis: {comb_df.shape[0]}")

with st.expander("Show Used Data", expanded=True):
    st.write("Used dataframe of stock and bond returns:")
    st.dataframe(comb_df)

with st.expander("Show Stock value development", expanded=True):
    plot_df(
        comb_df,
        title="Stock open prices",
        col_regexp_substring=r"open$",
        show_plot=False,
        show_plot_streamlit=True,
    )

    plot_df(
        comb_df,
        title="Selected stock returns",
        col_regexp_substring=return_col,
        show_plot=False,
        show_plot_streamlit=True,
    )


means, var, covs, corr = get_descriptive_statistics(
    comb_df,
    print_stats=False,
    start_date=start_date,
    end_date=end_date,
    col_regexp_substring=return_col,
)

st.write("Means of the selected stock returns:")
barplot_means(means, show_plot=False, show_plot_streamlit=True)

st.write("Variance and Correlations of the selected stock returns:")
plot_covariances_matrix(
    var=var,
    corr=corr,
    plot_variance_bar=True,
    plot_heatmap=True,
    show_plot=False,
    show_plot_streamlit=True,
)

st.write("Autocorrelation of the selected stock returns:")
with st.expander("Show Autocorrelation", expanded=False):
    st.markdown(
        """
        Autocorrelation describes the correlation of the
        time series of returns with a lagged version of itself.
        It shows a form of relationship between the returns
        in the consecutive days.
        """
    )
    plot_autocorrelation(
        comb_df,
        col_regexp_substring=return_col,
        show_plot=False,
        show_plot_streamlit=True,
    )

with st.expander("Show Missing and Zero Values", expanded=False):
    get_missing_and_zero_values(
        comb_df,
        get_missing=True,
        get_zero=True,
        print_results=False,
        print_result_streamlit=True,
        col_regexp_substring=return_col,
    )
