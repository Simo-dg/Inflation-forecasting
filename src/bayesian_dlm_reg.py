import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.utils import train_test_split_time_series

def dlm_model(X, y=None):
    T, P = X.shape

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Gaussian Random Walk increments for coefficients over time
    beta_increments = numpyro.sample("beta_increments", dist.Normal(0, 0.1).expand([T, P]))
    beta = numpyro.deterministic("beta", jnp.cumsum(beta_increments, axis=0))  # shape (T, P)

    mu = (X * beta).sum(axis=1)
    numpyro.deterministic("mu", mu)

    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def forecast_one_step(beta_prev_samples, X_next, sigma_samples, beta_noise_scale=0.1):
    """
    Forecast test horizon one step at a time using beta RW dynamics.
    beta_prev_samples: (S, P) array of beta samples at last train time step
    X_next: (H, P) test covariates scaled
    sigma_samples: (S,) noise scale samples from posterior
    """
    num_samples, P = beta_prev_samples.shape
    H = X_next.shape[0]
    forecasts = np.zeros((num_samples, H))

    beta_t = beta_prev_samples.copy()

    for t in range(H):
        # Random walk step for beta_t+1
        epsilon = np.random.normal(loc=0.0, scale=beta_noise_scale, size=(num_samples, P))
        beta_t = beta_t + epsilon

        # Compute mean prediction for time t
        mu_t = np.sum(beta_t * X_next[t], axis=1)  # (S,)
        
        # Add observation noise
        noise = np.random.normal(loc=0.0, scale=sigma_samples, size=num_samples)
        forecasts[:, t] = mu_t + noise

    return forecasts  # (S, H)


def run_dlm_dynamic_regression(df, target_col, train_start, train_end, test_start, test_end,
                               num_warmup=1500, num_samples=5000, num_chains=4):
    X_train, y_train, X_test, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]

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

    # Plot posterior mean coefficients over training time
    plt.figure(figsize=(12, 6))
    for j in range(mean_beta.shape[1]):
        plt.plot(X_train.index.to_numpy(), mean_beta[:, j], label=X_train.columns[j])
    plt.title("Posterior Mean of Time-Varying Coefficients (NumPyro)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_dynamic_betas_numpyro.png")
    plt.close()

    # Prepare test data
    X_test_scaled = scaler_X.transform(X_test)
    beta_last_samples = np.array(samples["beta"][:, -1, :])  # (S, P)
    sigma_samples = np.array(samples["sigma"])               # (S,)

    # One-step ahead forecast on test set with uncertainty
    forecasts_scaled = forecast_one_step(beta_last_samples, X_test_scaled, sigma_samples)

    # Convert forecasts back to original scale
    forecasts_original = scaler_y.inverse_transform(forecasts_scaled.T)  # shape (H, S)

    # Calculate mean and credible intervals
    mean_forecast = forecasts_original.mean(axis=1)
    lower_bound = np.percentile(forecasts_original, 5, axis=1)
    upper_bound = np.percentile(forecasts_original, 95, axis=1)

    # Plot forecast vs observed with credible intervals
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), mean_forecast, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.fill_between(y_test.index.to_numpy(), lower_bound, upper_bound, color="red", alpha=0.3, label="90% CI")
    plt.title(f"DLM regression (RMSE={rmse_test:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/dlm_regression_forecast.png")
    plt.close()

    # Compute RMSE on mean forecast
    rmse_test = np.sqrt(np.mean((mean_forecast - y_test.values) ** 2))
    print(f"NumPyro DLM One-Step Ahead RMSE (Test): {rmse_test:.4f}")

    return samples, rmse_test
