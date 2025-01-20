# pylint: disable=C0103
"""
Fourth non-entry page of the app.
Contains the Anomaly Detection of the stock and bond returns.
Utilizes autoencoder models for anomaly detection.
NOTE: The models need to be trained, which can take some time.
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
    from functions.anomaly_detection_functions import (
        plot_joint_autoencode_anomaly,
        plot_marginal_autoencode_anomaly,
        get_joint_autoencode_anomaly,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


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

comb_df.index = pd.to_datetime(comb_df.index)

st.title("Anomaly Detection")
st.subheader(
    "Perform anomaly detection on the stock and bond returns using autoencoder models"
)

st.markdown(
    """
    This page allows you to perform anomaly detection on
      the stock and bond returns using autoencoder models.
    The following functionalities are available on this page:
    
    1. **Marginal Anomaly Detection**: Perform anomaly detection on individual
      stocks by training separate autoencoder models for each stock.
    2. **Joint Anomaly Detection**: Perform anomaly detection on multiple
      stocks by training a single autoencoder model jointly on the stock returns.
    3. **Visualize Anomalies**: Visualize the detected anomalies on the stock returns.
    """
)

st.write(
    "Details on autoencoders can be found at [Wikipedia](https://en.wikipedia.org/wiki/Autoencoder)"
)

st.subheader("Marginal anomaly detection on stock data returns")
st.write(
    "Anomaly detection on single stock performed "
    "by training outoencoder model on the stock returns."
)
st.markdown(
    "*It can be seen that resulting anomalies are just outliers with appropriate threshold set.*"
)
st.markdown("**NOTE** Underlying models need to be trained, which can take some time.")

with st.expander("Marginal Anomaly Detection"):
    chosen_stocks = st.multiselect("Select stocks", tickers)

    ## Get only selected tickers from the comb_df
    remaining_cols = [
        col for col in comb_df.columns if col.split("_")[0] in chosen_stocks
    ]
    selected_comb_df = comb_df[remaining_cols]

    with st.spinner("Training models..."):
        plot_marginal_autoencode_anomaly(
            selected_comb_df,
            col_regexp_substring=return_col,
            show_plots=False,
            show_plots_streamlit=True,
        )


st.subheader("Joint anomaly detection on stock data returns")
st.write(
    "Anomaly detection on multiple stocks performed by training "
    "single outoencoder model jointly on the stock returns."
)
st.markdown(
    "*In this case, labeled anomalies are truly anomalies in sense "
    "of being hard to predict observations*"
)
st.markdown(
    "*Modeling stock returns jointly allows for capturing relationships "
    "between returns of individual stocks*"
)
st.markdown("**NOTE:** Underlying model needs to be trained, which can take some time.")

with st.expander("Joint Anomaly Detection"):
    with st.spinner("Training model..."):
        returns_data, anomaly_df = get_joint_autoencode_anomaly(
            comb_df, col_regexp_substring=return_col
        )
    plot_joint_autoencode_anomaly(
        returns_data, anomaly_df, show_plots=False, show_plots_streamlit=True
    )
