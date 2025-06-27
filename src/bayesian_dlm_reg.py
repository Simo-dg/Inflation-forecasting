import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import arviz as az
from src.utils import train_test_split_time_series

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


def run_dlm_dynamic_regression(df, target_col, train_start, train_end, test_start, test_end,
                               num_warmup=1500, num_samples=5000, num_chains=4):
    """ keep_cols = [
        "Unemployment - %",
        "Economic_sentiment",
        "Exchangerate_CPI",
        "10years_gov_bond_rate"
    ]

    X = df[keep_cols].copy()
    y = df[target_col].copy() """

    X_train, y_train, X_test, y_test = train_test_split_time_series(
            df, train_start, train_end, test_start, test_end, target_col
    )
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]

    """  X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X, y = X.align(y, join='inner', axis=0)
    X = X.dropna()
    y = y.loc[X.index] """

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    kernel = NUTS(dlm_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, X=jnp.array(X_train_scaled), y=jnp.array(y_train_scaled))

    samples = mcmc.get_samples()

    mean_beta = jnp.mean(samples["beta"], axis=0)  # (T, P)
    mean_mu = jnp.mean(samples["mu"], axis=0)      # (T,)

    mean_beta = np.array(mean_beta)
    mean_mu = np.array(mean_mu)

    # Plot coefficients over training time
    plt.figure(figsize=(12, 6))
    for j in range(mean_beta.shape[1]):
        plt.plot(X_train.index.to_numpy(), mean_beta[:, j], label=X_train.columns[j])
    plt.title("Posterior Mean of Time-Varying Coefficients (NumPyro)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_dynamic_betas_numpyro.png")
    plt.close()

    # Forecast on test set using last beta from training (simple approach)
    # Optionally, you can extend the model to forecast dynamically
    beta_last = mean_beta[-1, :]  # shape (P,)
    X_test_scaled = scaler_X.transform(X_test)
    mu_test = X_test_scaled @ beta_last

    # Scale back predictions to original scale
    mu_test_original = scaler_y.inverse_transform(mu_test.reshape(-1, 1)).flatten()

    # Calculate test RMSE
    residuals_test = mu_test_original - y_test.values
    rmse_test = np.sqrt(np.mean(residuals_test ** 2))

    # Plot forecast vs actual for test period
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed")
    plt.plot(y_test.index.to_numpy(), mu_test_original, label="DLM Prediction", linestyle="--")
    plt.title("Dynamic Linear Regression Forecast (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_dynamic_forecast_test_numpyro.png")
    plt.close()

    print(f"NumPyro DLM RMSE (Test): {rmse_test:.4f}")

    return samples, rmse_test