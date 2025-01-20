"""
Tests for the portfolio_functions module
"""

import sys
import os

from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib

# Due to unexpected import functionality, following lines are needed
# for all internal import.
# Pylint warning regarding duplicate code is silenced

# pylint: disable=R0801
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from functions.portfolio_functions import (
        portfolio_variance,
        portfolio_std,
        efficient_frontier_complete,
        efficient_frontier_complete_riskless,
        compute_markowitz_weights,
        compute_equal_weights_portfolio,
        compute_sharpe_nonnegative_weights_external,
        compute_max_sharpe_weights,
        compute_inverse_volatility_weighed_portfolio,
        compute_minimum_variance_portfolio,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Block plots from showing up
matplotlib.use("Agg")


def test_portfolio_variance(simple_weights, simple_cov_matrix):
    """Test the portfolio_variance function"""
    computed_variance = portfolio_variance(simple_weights, simple_cov_matrix)
    assert round(computed_variance, 3) == 0.075


def test_portfolio_std(simple_weights, simple_cov_matrix):
    """Test the portfolio_std function"""
    computed_std = portfolio_std(simple_weights, simple_cov_matrix)
    assert round(computed_std, 3) == 0.274


def test_efficient_frontier_complete(base_comb_means, base_comb_cov, base_comb_var):
    """Test the efficient_frontier_complete function"""

    efficient_results = efficient_frontier_complete(
        means=base_comb_means,
        covs=base_comb_cov,
        var=base_comb_var,
        plot_results=False,
        num_points=5,
    )

    assert len(efficient_results) == 5
    assert round(efficient_results[2][0], 5) == -0.00037
    assert round(efficient_results[2][1], 3) == 0.011


@patch("matplotlib.pyplot.show")
def test_efficient_frontier_complete_plot(
    mock_show, base_comb_means, base_comb_cov, base_comb_var
):
    """Test the efficient_frontier_complete function plot"""

    _ = efficient_frontier_complete(
        means=base_comb_means,
        covs=base_comb_cov,
        var=base_comb_var,
        plot_results=True,
        num_points=5,
    )

    assert mock_show.called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Efficient Frontier (Without Risk-Free Asset)"


def test_efficient_frontier_complete_riskless(
    base_comb_means, base_comb_cov, base_comb_var
):
    """Test the efficient_frontier_complete_riskless function"""
    weights_tangent, mean_tangent, std_tangent = efficient_frontier_complete_riskless(
        means=base_comb_means,
        covs=base_comb_cov,
        var=base_comb_var,
        riskless_rate=0.0,
        plot_results=False,
    )

    assert round(weights_tangent[0], 3) == 0.536
    assert round(mean_tangent, 3) == 0.004
    assert round(std_tangent, 3) == 0.033


@patch("matplotlib.pyplot.show")
def test_efficient_frontier_complete_riskless_plot(
    mock_show, base_comb_means, base_comb_cov, base_comb_var
):
    """Test the efficient_frontier_complete_riskless function plot"""

    _ = efficient_frontier_complete_riskless(
        means=base_comb_means,
        covs=base_comb_cov,
        var=base_comb_var,
        riskless_rate=0.0,
        plot_results=True,
    )

    assert mock_show.called_once()

    fig = plt.gcf()
    ax = fig.gca()
    assert ax.get_title() == "Efficient Frontier (With Risk-Free Asset)"


def test_compute_markowitz_weights(simple_means, simple_cov_matrix):
    """Test the compute_markowitz_weights function"""
    share_weights, riskless_weight = compute_markowitz_weights(
        means=simple_means,
        covs=simple_cov_matrix,
        riskless_rate=0.0,
        required_return=0.1,
    )

    assert round(share_weights[0], 3) == 0.0
    assert round(share_weights[1], 3) == 0.5
    assert riskless_weight == 0.5


def test_compute_equal_weights_portfolio(simple_means):
    """Test the compute_equal_weights_portfolio function"""
    share_weights, riskless_weight = compute_equal_weights_portfolio(
        simple_means, use_riskless_asset=True
    )

    assert round(share_weights[0], 3) == 0.333
    assert round(share_weights[1], 3) == 0.333
    assert round(riskless_weight, 3) == 0.333


def test_compute_sharpe_nonnegative_weights_external(simple_means, simple_cov_matrix):
    """Test the compute_sharpe_nonnegative_weights_external function"""
    share_weights = compute_sharpe_nonnegative_weights_external(
        means=simple_means, covs=simple_cov_matrix, riskless_rate=0.0
    )

    assert round(share_weights[0], 3) == 0.0
    assert round(share_weights[1], 3) == 1.0


def test_compute_max_sharpe_weights(simple_means, simple_cov_matrix):
    """Test the compute_max_sharpe_weights function"""
    share_weights = compute_max_sharpe_weights(
        means=simple_means, covs=simple_cov_matrix, riskless_rate=0.0
    )

    assert round(share_weights[0], 3) == 0.0
    assert round(share_weights[1], 3) == 1.0


def test_compute_inverse_volatility_weighed_portfolio(simple_var):
    """Test the compute_inverse_volatility_weighed_portfolio function"""
    share_weights = compute_inverse_volatility_weighed_portfolio(simple_var)

    assert round(share_weights[0], 3) == 0.5
    assert round(share_weights[1], 3) == 0.5


def test_compute_minimum_variance_portfolio(simple_means, simple_cov_matrix):
    """Test the compute_minimum_variance_portfolio function"""
    share_weights = compute_minimum_variance_portfolio(simple_means, simple_cov_matrix)

    assert round(share_weights[0], 3) == 0.5
    assert round(share_weights[1], 3) == 0.5
