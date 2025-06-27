import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from src.utils import train_test_split_time_series  # your utility for splitting

def run_dlm_pymc(df, target_col, train_start, train_end, test_start, test_end):
    # Split data into train/test using your utility
    _, y_train, _, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )
    
    # Fill missing values in training target if any (or interpolate)
    y = y_train.copy()
    y = y.fillna(method='ffill').fillna(method='bfill')  # forward fill then backfill if needed

    T = len(y)

    with pm.Model() as dlm_model:
        sigma = pm.Exponential("sigma", 1.0)
        # Use smaller sigma for better sampler stability
        beta = pm.GaussianRandomWalk("beta", sigma=0.1, shape=T)
        y_obs = pm.Normal("y_obs", mu=beta, sigma=sigma, observed=y.values)

        trace = pm.sample(
            draws=10000, tune=3000, target_accept=0.99, random_seed=42, max_treedepth=15
        )
    
    # Diagnostic plots
    az.plot_pair(trace, var_names=["beta", "sigma"], kind="kde", divergences=True)
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_pairplot.png")
    plt.close()

    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_trace.png")
    plt.close()

    # Posterior mean estimate of beta
    mean_beta = trace.posterior["beta"].mean(dim=("chain", "draw")).values.flatten()

    if not np.all(np.isfinite(mean_beta)):
        print("Warning: Non-finite values in mean_beta. Check model convergence.")
        mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0)

    # Plot training observed vs filtered beta, plus test observed (no forecast)
    plt.figure(figsize=(10, 4))
    plt.plot(y.index.to_numpy(), y.values, label="Observed (Train)")
    plt.plot(y.index.to_numpy(), mean_beta, label="Filtered βₜ (Train)", linestyle="--")
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", alpha=0.7)
    plt.title("DLM (PyMC) Train Observed and Filtered")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_train_forecast.png")
    plt.close()

    # Compute RMSE on train data only (since no prediction on test)
    rmse = np.sqrt(np.mean((mean_beta - y.values) ** 2))
    
    return trace, rmse
