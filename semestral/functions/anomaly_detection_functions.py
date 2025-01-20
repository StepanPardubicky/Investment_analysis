# pylint: disable=import-error,no-name-in-module
"""
Anomaly detection functions
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import MinMaxScaler


# Following import work fine, but pylint incorrectly reports import errors
from tensorflow.keras.models import (
    Model,
)
from tensorflow.keras.layers import (
    Input,
    Dense,
)
from tensorflow.keras.optimizers import (
    Adam,
)

# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Workaround wierd python relative import behavior and pylint warnings
try:
    from functions.data_manipulation_functions import select_relevant_cols
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


default_autoencoder_kwargs = {
    "hidden_neuron_list": [10, 2],
    "contamination": 0.05,
    "epoch_num": 20,
}


def _anomaly_detection_autoencode_single_stock(
    stock_returns: pd.Series, autoencoder_kwargs: dict | None = None
):
    """
    Autoencoder anomaly detection for time serie of single stock returns
    Core of the function coauthored by GPT
    """

    if autoencoder_kwargs is None:
        autoencoder_kwargs = default_autoencoder_kwargs

    returns = stock_returns.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(returns)

    model = AutoEncoder(**autoencoder_kwargs)

    model.fit(scaled_returns)

    anomaly_labels = model.predict(scaled_returns)

    return anomaly_labels


def plot_marginal_autoencode_anomaly(
    df: pd.DataFrame,
    col_regexp_substring: str = r"interdaily_pct_change$",
    show_plots: bool = True,
    show_plots_streamlit: bool = False,
    autoencoder_kwargs: dict | None = None,
):
    """
    Plots the marginal anomaly scores for the autoencoder model
    """

    if autoencoder_kwargs is None:
        autoencoder_kwargs = default_autoencoder_kwargs

    selected_df = select_relevant_cols(df, col_regexp_substring=col_regexp_substring)

    for col in selected_df.columns:

        anomaly_labels = _anomaly_detection_autoencode_single_stock(
            selected_df[col], autoencoder_kwargs=autoencoder_kwargs
        )

        fig, ax = plt.subplots()
        ax.plot(selected_df.index, selected_df[col], label="Anomaly Score")
        ax.scatter(
            selected_df.index,
            selected_df[col],
            c=anomaly_labels,
            label="Non-Anomalous",
        )
        ax.set_title(f"Marginal Detected Anomalies for {col}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Return")
        ax.legend()

        if show_plots:
            plt.show()
        if show_plots_streamlit:
            st.pyplot(fig)


def _build_autoencoder_model(
    scaled_returns: np.ndarray, model_input_dim: int, **kwargs
):
    """
    Build autoencoder model
    Inner autoencoder training functionality was coauthored by GPT
    """
    # Handle model kwargs
    encoding_dim = kwargs.get("encoding_dim", 2)
    hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (32, 16))
    activation = kwargs.get("activation", "relu")
    num_epochs = kwargs.get("num_epochs", 20)
    batch_size = kwargs.get("batch_size", 32)

    # Define Model (use keras functional API)

    input_layer = Input(shape=(model_input_dim,))

    # Encoding layers
    encoded = Dense(hidden_layer_sizes[0], activation=activation)(input_layer)
    encoded = Dense(hidden_layer_sizes[1], activation=activation)(encoded)
    encoded = Dense(encoding_dim, activation=activation)(encoded)

    # Decoding layers
    decoded = Dense(hidden_layer_sizes[1], activation=activation)(encoded)
    decoded = Dense(hidden_layer_sizes[0], activation=activation)(decoded)
    decoded = Dense(model_input_dim, activation="linear")(decoded)

    # Autoencoder model
    autoencoder = Model(input_layer, decoded)

    # Compile model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train model
    autoencoder.fit(
        scaled_returns,
        scaled_returns,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
    )

    return autoencoder


def get_joint_autoencode_anomaly(
    df: pd.DataFrame, col_regexp_substring: str = r"interdaily_pct_change$", **kwargs
) -> pd.DataFrame:
    """
    Train Autoencoder model on the returns data and return anomaly labels
    """

    returns_data = select_relevant_cols(df, col_regexp_substring=col_regexp_substring)

    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    model_input_dim = returns_data.shape[1]

    # Define Model (use keras functional API)

    autoencoder = _build_autoencoder_model(
        scaled_returns,
        model_input_dim,
        **kwargs,
    )

    # input_layer = Input(shape=(model_input_dim,))

    # # Encoding layers
    # encoded = Dense(hidden_layer_sizes[0], activation=activation)(input_layer)
    # encoded = Dense(hidden_layer_sizes[1], activation=activation)(encoded)
    # encoded = Dense(encoding_dim, activation=activation)(encoded)

    # # Decoding layers
    # decoded = Dense(hidden_layer_sizes[1], activation=activation)(encoded)
    # decoded = Dense(hidden_layer_sizes[0], activation=activation)(decoded)
    # decoded = Dense(model_input_dim, activation="linear")(decoded)

    # # Autoencoder model
    # autoencoder = Model(input_layer, decoded)

    # # Compile model
    # autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # # Train model
    # autoencoder.fit(
    #     scaled_returns,
    #     scaled_returns,
    #     epochs=num_epochs,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    # Use predictions to get anomaly labels

    reconstructed_data = autoencoder.predict(scaled_returns)

    returns_reconstruction_error = np.abs(scaled_returns - reconstructed_data)

    error_threshold = np.percentile(
        returns_reconstruction_error,
        95,
    )
    detected_anomalies = returns_reconstruction_error > error_threshold

    anomaly_df = pd.DataFrame(
        detected_anomalies, columns=returns_data.columns, index=returns_data.index
    )

    return returns_data, anomaly_df


def plot_joint_autoencode_anomaly(
    returns_data: pd.DataFrame, anomaly_df: pd.DataFrame, **kwargs
):
    """
    Plots the joint anomaly scores for the autoencoder model
    """

    # Handle kwargs
    show_plots = kwargs.get("show_plots", True)
    show_plots_streamlit = kwargs.get("show_plots_streamlit", False)

    for col in returns_data.columns:

        anomaly_labels = anomaly_df[col]

        fig, ax = plt.subplots()
        ax.plot(returns_data.index, returns_data[col], label="Returns")
        ax.scatter(
            returns_data.index,
            returns_data[col],
            c=anomaly_labels,
            label="Anomaly/Normal",
        )
        ax.set_title(f"Marginal Detected Anomalies for {col}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Return")
        ax.legend()

        if show_plots:
            plt.show()
        if show_plots_streamlit:
            st.pyplot(fig)
