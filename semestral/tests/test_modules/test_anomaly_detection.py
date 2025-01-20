"""
Tests for the anomaly_detection_functions file.
"""

import sys
import os
import random
from unittest.mock import patch

# import pytest

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf

# Following pylint error is false positive
# pylint: disable=import-error, no-name-in-module
from tensorflow.keras.models import Model


# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.anomaly_detection_functions import (
        _anomaly_detection_autoencode_single_stock,
        plot_marginal_autoencode_anomaly,
        _build_autoencoder_model,
        get_joint_autoencode_anomaly,
        plot_joint_autoencode_anomaly,
    )
    from functions.data_manipulation_functions import filter_by_date
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Set random seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Block plots from showing up
matplotlib.use("Agg")


def test_anomaly_detection_autoencode_single_stock(base_comb_df):
    """Test for the _anomaly_detection_autoencode_single_stock function"""
    sub_df = filter_by_date(base_comb_df, "2022-01-03", "2022-04-01")
    anomaly_labels = _anomaly_detection_autoencode_single_stock(
        stock_returns=sub_df["COST_interdaily_pct_change"],
        autoencoder_kwargs={
            "hidden_neuron_list": [3, 1],
            "contamination": 0.5,
            "epoch_num": 1,
        },
    )

    assert isinstance(anomaly_labels, np.ndarray)

    assert anomaly_labels[0] == 0
    assert anomaly_labels[1] == 0


@patch("matplotlib.pyplot.show")
def test_plot_marginal_autoencode_anomaly(mock_show, base_comb_df):
    """Test for the plot_marginal_autoencode_anomaly function"""
    sub_df = filter_by_date(base_comb_df, "2022-01-03", "2022-04-01")
    plot_marginal_autoencode_anomaly(
        df=sub_df,
        col_regexp_substring=r"COST_interdaily_pct_change$",  ## Select on single stock for testing
        show_plots=True,
        show_plots_streamlit=False,
        autoencoder_kwargs={
            "hidden_neuron_list": [3, 1],
            "contamination": 0.5,
            "epoch_num": 1,
        },
    )

    assert mock_show.call_count == 1


def test_build_autoencoder_model(anomaly_sub_df):
    """Test for the _build_autoencoder_model function"""
    model = _build_autoencoder_model(
        scaled_returns=anomaly_sub_df,  # Are not scaled for the test, shouldnt matter
        model_input_dim=2,
        encoding_dim=1,
        hidden_layer_sizes=(8, 2),
        activation="relu",
        num_epochs=1,
        batch_size=6,
    )

    assert isinstance(model, Model)


def test_get_joint_autoencode_anomaly(anomaly_sub_df):
    """Test for the get_joint_autoencode_anomaly function"""
    returns_data, anomaly_df = get_joint_autoencode_anomaly(
        df=anomaly_sub_df,
        col_regexp_substring=r"interdaily_pct_change$",
        encoding_dim=1,
        hidden_layer_sizes=(8, 2),
        activation="relu",
        num_epochs=1,
        batch_size=6,
    )

    assert isinstance(returns_data, pd.DataFrame)
    assert isinstance(anomaly_df, pd.DataFrame)

    assert anomaly_df.iloc[0, 0] == 0


@patch("matplotlib.pyplot.show")
def test_plot_joint_autoencode_anomaly(
    mock_show, anomaly_test_get_joint_autoencode_results
):
    """Test for the plot_joint_autoencode_anomaly function"""
    plot_joint_autoencode_anomaly(
        *anomaly_test_get_joint_autoencode_results,
        show_plots=True,
        show_plots_streamlit=False,
    )
    assert mock_show.call_count == 2

    fig = plt.gcf()
    ax = fig.gca()
    assert (
        "Marginal Detected Anomalies for ANSS_interdaily_pct_change" in ax.get_title()
    )
    plt.close()
