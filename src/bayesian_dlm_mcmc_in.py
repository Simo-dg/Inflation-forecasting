import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

def run_dlm_pymc_in(df, target_col):
    y = df[target_col].dropna()
    T = len(y)

    with pm.Model() as dlm_model:
        sigma = pm.Exponential("sigma", 1.0)
        beta = pm.GaussianRandomWalk("beta", sigma=0.5, shape=T)
        y_obs = pm.Normal("y_obs", mu=beta, sigma=sigma, observed=y.values)

        trace = pm.sample(
            draws=10000, tune=1500, target_accept=0.99, random_seed=42
        )
    az.plot_pair(trace, var_names=["beta", "sigma"], kind="kde", divergences=True)
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_pairplot.png")
    plt.close()

    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_trace.png")
    plt.close()

    # Extract posterior mean for beta
    # Convert to numpy to avoid pandas multidim indexing issues
    mean_beta = trace.posterior["beta"].mean(dim=("chain", "draw")).values.flatten()

    # Ensure no NaNs/Infs in mean_beta before plotting
    if not np.all(np.isfinite(mean_beta)):
        print("Warning: Non-finite values in mean_beta. Check model convergence.")
        mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0)

    plt.figure(figsize=(10, 4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label="Observed")
    plt.plot(y.index.to_numpy(), mean_beta, label="Filtered βₜ", linestyle="--")
    plt.title("DLM (PyMC) Forecast vs Observed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_forecast.png")
    plt.close()

    rmse = np.sqrt(np.mean((mean_beta - y.values)**2))
    return trace, rmse
