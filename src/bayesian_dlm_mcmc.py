import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from src.utils import train_test_split_time_series

def run_dlm_pymc(df, target_col, train_start, train_end, test_start, test_end, num_particles=1000):
    # Split data
    _, y_train, _, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )

    # Fill missing in training target (if any)
    y = y_train.copy().fillna(method='ffill').fillna(method='bfill')
    T_train = len(y)
    T_test = len(y_test)

    with pm.Model() as dlm_model:
        sigma = pm.Exponential("sigma", 1.0)
        beta = pm.GaussianRandomWalk("beta", sigma=0.1, shape=T_train)
        y_obs = pm.Normal("y_obs", mu=beta, sigma=sigma, observed=y.values)

        trace = pm.sample(
            draws=2000, tune=1000, target_accept=0.95, random_seed=42, max_treedepth=15, cores=4, chains=4
        )

    # Extract posterior samples for beta and sigma
    beta_samples = trace.posterior["beta"].stack(sample=("chain", "draw")).values  # shape (T_train, n_samples)
    sigma_samples = trace.posterior["sigma"].stack(sample=("chain", "draw")).values  # shape (n_samples,)

    n_samples = beta_samples.shape[1]

    # Initialize particles at last training time as posterior samples of beta[T_train-1]
    particles = beta_samples[-1, :].copy()  # shape (n_samples,)

    # Sigma samples repeated for particles (some randomization possible here)
    sigma_particles = sigma_samples.copy()

    # If you want exactly num_particles, resample posterior samples (with replacement)
    if n_samples > num_particles:
        idx = np.random.choice(n_samples, num_particles, replace=False)
        particles = particles[idx]
        sigma_particles = sigma_particles[idx]
        n_samples = num_particles
    else:
        num_particles = n_samples  # adjust

    forecast_means = []
    forecast_lower = []
    forecast_upper = []

    weights = np.ones(n_samples) / n_samples

    # One-step ahead forecasting loop over test points
    for t in range(T_test):
        # Propagate particles by one step of Gaussian RW: beta_t = beta_{t-1} + Normal(0, sigma)
        propagated_particles = particles + np.random.normal(0, sigma_particles)

        # Predicted observation mean = propagated_particles
        # Observation likelihood under Normal with sigma_particles
        obs = y_test.values[t]

        log_weights = -0.5 * np.log(2 * np.pi * sigma_particles ** 2) - 0.5 * ((obs - propagated_particles) ** 2) / (sigma_particles ** 2)
        # Stabilize weights
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Weighted forecast mean and credible intervals
        mean_forecast = np.sum(propagated_particles * weights)
        lower_forecast = np.percentile(propagated_particles, 5)
        upper_forecast = np.percentile(propagated_particles, 95)

        forecast_means.append(mean_forecast)
        forecast_lower.append(lower_forecast)
        forecast_upper.append(upper_forecast)

        # Resample particles if effective sample size is low
        ess = 1. / np.sum(weights ** 2)
        if ess < n_samples / 2:
            indices = np.random.choice(n_samples, n_samples, p=weights)
            propagated_particles = propagated_particles[indices]
            sigma_particles = sigma_particles[indices]
            weights = np.ones(n_samples) / n_samples

        # Update particles for next iteration
        particles = propagated_particles

    forecast_means = np.array(forecast_means)
    forecast_lower = np.array(forecast_lower)
    forecast_upper = np.array(forecast_upper)

    rmse_test = np.sqrt(np.mean((forecast_means - y_test.values) ** 2))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecast_means, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.fill_between(y_test.index.to_numpy(), forecast_lower, forecast_upper, color="red", alpha=0.3, label="90% CI")
    plt.title(f"DLM (PyMC) One-step Ahead Forecast (RMSE={rmse_test:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pymc_dlm_one_step_forecast_test_only.png")
    plt.close()

    return trace, rmse_test
