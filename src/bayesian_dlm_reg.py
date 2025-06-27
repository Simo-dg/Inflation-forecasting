import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import arviz as az

def dlm_model(X, y=None):
    T, P = X.shape

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Gaussian Random Walk for each coefficient over time
    # We'll model increments ~ Normal(0, 0.1), then cumulative sum to get beta_t
    beta_increments = numpyro.sample("beta_increments", dist.Normal(0, 0.1).expand([T, P]))
    beta = numpyro.deterministic("beta", jnp.cumsum(beta_increments, axis=0))  # shape (T, P)

    mu = (X * beta).sum(axis=1)
    numpyro.deterministic("mu", mu)

    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def run_dlm_dynamic_regression(df, target_col, num_warmup=1500, num_samples=5000, num_chains=4):
    """ keep_cols = [
        "Unemployment - %",
        "Economic_sentiment",
        "Exchangerate_CPI",
        "10years_gov_bond_rate"
    ]

    X = df[keep_cols].copy()
    y = df[target_col].copy() """

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X, y = X.align(y, join='inner', axis=0)
    X = X.dropna()
    y = y.loc[X.index]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    kernel = NUTS(dlm_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, X=jnp.array(X_scaled), y=jnp.array(y_scaled))

    samples = mcmc.get_samples()

    mean_beta = jnp.mean(samples["beta"], axis=0)  # (T, P)
    mean_mu = jnp.mean(samples["mu"], axis=0)  # (T,)

    # Convert back to numpy
    mean_beta = np.array(mean_beta)
    mean_mu = np.array(mean_mu)

    time_index = y.index.to_numpy()

    # Plot posterior means of beta_t
    plt.figure(figsize=(12, 6))
    for j in range(mean_beta.shape[1]):
        plt.plot(time_index, mean_beta[:, j], label=X.columns[j])
    plt.title("Posterior Mean of Time-Varying Coefficients (NumPyro)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_dynamic_betas_numpyro.png")
    plt.close()

    # Forecast vs Actual
    plt.figure(figsize=(10, 4))
    plt.plot(time_index, y_scaled, label="Observed (scaled)")
    plt.plot(time_index, mean_mu, label="DLM Prediction (scaled)", linestyle="--")
    plt.title("Dynamic Linear Regression Forecast (NumPyro)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_dynamic_forecast_numpyro.png")
    plt.close()

    residuals = mean_mu - y_scaled
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"NumPyro DLM RMSE (scaled): {rmse:.4f}")

    return samples, rmse
