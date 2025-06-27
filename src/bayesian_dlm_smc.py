import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from tqdm import tqdm
from src.utils import train_test_split_time_series

def stratified_resample(weights, rng_key):
    N = len(weights)
    positions = (jnp.arange(N) + random.uniform(rng_key, shape=(N,))) / N
    cumsum = jnp.cumsum(weights)
    return jnp.searchsorted(cumsum, positions)

def particle_filter_dlm(y, time_index, num_particles=2000, lag=5):
    T = len(y)
    rng_key = random.PRNGKey(0)

    # Priors for noise variances (log-normal)
    log_sigma_obs_prior_mean = jnp.log(jnp.clip(jnp.nanstd(y) / 10, 0.01, 0.5))
    log_sigma_state_prior_mean = jnp.log(jnp.clip(jnp.nanstd(y) / 50, 0.001, 0.2))

    # Transition probabilities for regime switching
    p_stay_calm = 0.97
    p_stay_crisis = 0.95

    # Initialize particles and weights
    particles_level = jnp.zeros((num_particles, T))
    particles_trend = jnp.zeros((num_particles, T))
    particles_regime = jnp.zeros((num_particles, T), dtype=jnp.int32)
    particles_log_sigma_obs = jnp.ones((num_particles, T)) * log_sigma_obs_prior_mean
    particles_log_sigma_state = jnp.ones((num_particles, T)) * log_sigma_state_prior_mean

    weights = jnp.ones(num_particles) / num_particles

    rng_key, sk1, sk2 = random.split(rng_key, 3)
    particles_level = particles_level.at[:, 0].set(
        y[0] + random.normal(sk1, (num_particles,)) * jnp.exp(log_sigma_obs_prior_mean))
    particles_trend = particles_trend.at[:, 0].set(random.normal(sk2, (num_particles,)) * 0.1)
    particles_regime = particles_regime.at[:, 0].set(0)
    particles_log_sigma_obs = particles_log_sigma_obs.at[:, 0].set(log_sigma_obs_prior_mean)
    particles_log_sigma_state = particles_log_sigma_state.at[:, 0].set(log_sigma_state_prior_mean)

    filtered_level = jnp.zeros(T)
    filtered_trend = jnp.zeros(T)
    filtered_regime_prob = jnp.zeros(T)

    jitter_std = 0.02
    transition_matrix = jnp.array([[p_stay_calm, 1 - p_stay_calm],
                                   [1 - p_stay_crisis, p_stay_crisis]])

    for t in tqdm(range(1, T), desc="Particle Filtering"):
        ess = 1. / jnp.sum(weights ** 2)
        if ess < num_particles / 2:
            rng_key, subkey = random.split(rng_key)
            idx = stratified_resample(weights, subkey)
            particles_level = particles_level[idx]
            particles_trend = particles_trend[idx]
            particles_regime = particles_regime[idx]
            particles_log_sigma_obs = particles_log_sigma_obs[idx]
            particles_log_sigma_state = particles_log_sigma_state[idx]
            weights = jnp.ones(num_particles) / num_particles

            rng_key, jkey = random.split(rng_key)
            particles_level = particles_level.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std)
            rng_key, jkey = random.split(rng_key)
            particles_trend = particles_trend.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std)
            rng_key, jkey = random.split(rng_key)
            particles_log_sigma_obs = particles_log_sigma_obs.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std / 5)
            rng_key, jkey = random.split(rng_key)
            particles_log_sigma_state = particles_log_sigma_state.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std / 5)

        regime_prev = particles_regime[:, t - 1]
        probs = transition_matrix[regime_prev]

        rng_key, subkey = random.split(rng_key)
        rng_keys_regime = random.split(subkey, num_particles)
        sample_one = lambda rng_key, p: dist.Categorical(probs=p).sample(rng_key)
        regime_t = jax.vmap(sample_one)(rng_keys_regime, probs)
        particles_regime = particles_regime.at[:, t].set(regime_t)

        rng_key, sk_obs, sk_state = random.split(rng_key, 3)
        log_sigma_obs_t = particles_log_sigma_obs[:, t - 1] + random.normal(sk_obs, (num_particles,)) * 0.05
        log_sigma_state_t = particles_log_sigma_state[:, t - 1] + random.normal(sk_state, (num_particles,)) * 0.05

        log_sigma_obs_t = jnp.clip(log_sigma_obs_t, jnp.log(0.005), jnp.log(1.0))
        log_sigma_state_t = jnp.clip(log_sigma_state_t, jnp.log(0.001), jnp.log(0.5))

        particles_log_sigma_obs = particles_log_sigma_obs.at[:, t].set(log_sigma_obs_t)
        particles_log_sigma_state = particles_log_sigma_state.at[:, t].set(log_sigma_state_t)

        sigma_obs_t = jnp.exp(log_sigma_obs_t)
        sigma_state_t = jnp.exp(log_sigma_state_t)

        sigma_obs_t = jnp.where(regime_t == 1, sigma_obs_t * 3.0, sigma_obs_t)
        sigma_state_t = jnp.where(regime_t == 1, sigma_state_t * 3.0, sigma_state_t)

        rng_key, sk1, sk2 = random.split(rng_key, 3)
        new_trend = 0.85 * jnp.tanh(particles_trend[:, t - 1]) + random.normal(sk1, (num_particles,)) * sigma_state_t
        new_level = particles_level[:, t - 1] + new_trend + random.normal(sk2, (num_particles,)) * sigma_state_t

        particles_trend = particles_trend.at[:, t].set(new_trend)
        particles_level = particles_level.at[:, t].set(new_level)

        log_weights = dist.Normal(new_level, sigma_obs_t).log_prob(y[t])
        log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
        log_weights -= jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weights = jnp.clip(weights, 1e-12, jnp.inf)
        weights /= jnp.sum(weights)

        filtered_level = filtered_level.at[t].set(jnp.sum(particles_level[:, t] * weights))
        filtered_trend = filtered_trend.at[t].set(jnp.sum(particles_trend[:, t] * weights))
        filtered_regime_prob = filtered_regime_prob.at[t].set(jnp.sum(weights * regime_t))

    beta_lower = jnp.percentile(particles_level, 5, axis=0)
    beta_upper = jnp.percentile(particles_level, 95, axis=0)
    beta_mean = jnp.mean(particles_level, axis=0)

    return filtered_level, beta_mean, beta_lower, beta_upper, particles_level, particles_trend, particles_regime, particles_log_sigma_obs[:, -1], particles_log_sigma_state[:, -1], weights, rng_key

def run_dlm_numpyro(df, target_col, train_start="2000-01-01", train_end="2019-12-31", 
                   test_start="2020-01-01", test_end=None, num_particles=5000, lag=5):
    if test_end is None:
        test_end = df.index.max().strftime("%Y-%m-%d")

    # Split data into train and test
    _, y_train, _, y_test = train_test_split_time_series(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        target_col=target_col
    )

    y_train_np = y_train.dropna().to_numpy()
    train_time_index = y_train.dropna().index

    # Run particle filter on training data
    filtered_level, beta_mean, beta_lower, beta_upper, particles_level, particles_trend, particles_regime, last_log_sigma_obs, last_log_sigma_state, weights, rng_key = particle_filter_dlm(
        y_train_np, train_time_index, num_particles=num_particles, lag=lag
    )

    # Prepare test data for incremental one-step ahead forecasting
    y_test_np = y_test.dropna().to_numpy()
    test_time_index = y_test.dropna().index
    T_test = len(y_test_np)

    # Initialize particles at test start from last train state (last time step)
    particles_level_test = particles_level[:, -1]
    particles_trend_test = particles_trend[:, -1]
    particles_regime_test = particles_regime[:, -1]
    particles_log_sigma_obs_test = last_log_sigma_obs
    particles_log_sigma_state_test = last_log_sigma_state
    weights_test = weights

    filtered_level_test = []
    forecast_mean_test = []
    forecast_lower_test = []
    forecast_upper_test = []

    jitter_std = 0.02
    transition_matrix = jnp.array([[0.97, 0.03],
                                   [0.05, 0.95]])

    # Rolling one-step ahead forecasting for test period
    for t in tqdm(range(T_test), desc="One-step ahead forecasting on test"):
        # 1-step ahead forecast: propagate particles without observing y[t]
        # Propagate regime
        probs = transition_matrix[particles_regime_test]
        rng_key, subkey = random.split(rng_key)
        rng_keys_regime = random.split(subkey, num_particles)
        sample_one = lambda rng_key, p: dist.Categorical(probs=p).sample(rng_key)
        forecast_regime = jax.vmap(sample_one)(rng_keys_regime, probs)

        # Propagate log sigma parameters with jitter
        rng_key, sk_obs, sk_state = random.split(rng_key, 3)
        forecast_log_sigma_obs = particles_log_sigma_obs_test + random.normal(sk_obs, (num_particles,)) * 0.05
        forecast_log_sigma_state = particles_log_sigma_state_test + random.normal(sk_state, (num_particles,)) * 0.05
        forecast_log_sigma_obs = jnp.clip(forecast_log_sigma_obs, jnp.log(0.005), jnp.log(1.0))
        forecast_log_sigma_state = jnp.clip(forecast_log_sigma_state, jnp.log(0.001), jnp.log(0.5))

        sigma_obs_t = jnp.exp(forecast_log_sigma_obs)
        sigma_state_t = jnp.exp(forecast_log_sigma_state)

        sigma_obs_t = jnp.where(forecast_regime == 1, sigma_obs_t * 3.0, sigma_obs_t)
        sigma_state_t = jnp.where(forecast_regime == 1, sigma_state_t * 3.0, sigma_state_t)

        # Propagate trend and level
        rng_key, sk1, sk2 = random.split(rng_key, 3)
        forecast_trend = 0.85 * jnp.tanh(particles_trend_test) + random.normal(sk1, (num_particles,)) * sigma_state_t
        forecast_level = particles_level_test + forecast_trend + random.normal(sk2, (num_particles,)) * sigma_state_t

        # Store forecast mean and credible intervals BEFORE update with y[t]
        mean_forecast = jnp.mean(forecast_level)
        lower_forecast = jnp.percentile(forecast_level, 5)
        upper_forecast = jnp.percentile(forecast_level, 95)

        forecast_mean_test.append(mean_forecast)
        forecast_lower_test.append(lower_forecast)
        forecast_upper_test.append(upper_forecast)

        # Now update weights with observation y[t]
        log_weights = dist.Normal(forecast_level, sigma_obs_t).log_prob(y_test_np[t])
        log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
        log_weights -= jnp.max(log_weights)
        weights_test = jnp.exp(log_weights)
        weights_test = jnp.clip(weights_test, 1e-12, jnp.inf)
        weights_test /= jnp.sum(weights_test)

        # Compute ESS and resample if needed
        ess = 1. / jnp.sum(weights_test ** 2)
        if ess < num_particles / 2:
            rng_key, subkey = random.split(rng_key)
            idx = stratified_resample(weights_test, subkey)
            forecast_level = forecast_level[idx]
            forecast_trend = forecast_trend[idx]
            forecast_regime = forecast_regime[idx]
            forecast_log_sigma_obs = forecast_log_sigma_obs[idx]
            forecast_log_sigma_state = forecast_log_sigma_state[idx]
            weights_test = jnp.ones(num_particles) / num_particles

            rng_key, jkey = random.split(rng_key)
            forecast_level = forecast_level + random.normal(jkey, (num_particles,)) * jitter_std
            rng_key, jkey = random.split(rng_key)
            forecast_trend = forecast_trend + random.normal(jkey, (num_particles,)) * jitter_std
            rng_key, jkey = random.split(rng_key)
            forecast_log_sigma_obs = forecast_log_sigma_obs + random.normal(jkey, (num_particles,)) * jitter_std / 5
            rng_key, jkey = random.split(rng_key)
            forecast_log_sigma_state = forecast_log_sigma_state + random.normal(jkey, (num_particles,)) * jitter_std / 5

        # Update particles for next iteration
        particles_level_test = forecast_level
        particles_trend_test = forecast_trend
        particles_regime_test = forecast_regime
        particles_log_sigma_obs_test = forecast_log_sigma_obs
        particles_log_sigma_state_test = forecast_log_sigma_state

        filtered_level_test.append(jnp.sum(particles_level_test * weights_test))

    forecast_mean_test = jnp.array(forecast_mean_test)
    forecast_lower_test = jnp.array(forecast_lower_test)
    forecast_upper_test = jnp.array(forecast_upper_test)
    filtered_level_test = jnp.array(filtered_level_test)

    rmse_test = float(np.sqrt(np.mean((y_test_np - forecast_mean_test) ** 2)))

    plt.figure(figsize=(12, 6))
    plt.plot(test_time_index.to_numpy(), y_test_np, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(test_time_index.to_numpy(), forecast_mean_test, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.fill_between(test_time_index.to_numpy(), forecast_lower_test, forecast_upper_test, color="red", alpha=0.3, label="90% CI")
    plt.title(f"Dynamic Linear Model Forecast (RMSE={rmse_test:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.savefig("results/dlm_forecast_test_numpyro.png")
    plt.close()

    return forecast_level, rmse_test

