import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from tqdm import tqdm

# Stratified resampling
def stratified_resample(weights, rng_key):
    N = len(weights)
    positions = (jnp.arange(N) + random.uniform(rng_key, shape=(N,))) / N
    cumsum = jnp.cumsum(weights)
    return jnp.searchsorted(cumsum, positions)

# Particle filter + smoother for DLM with latent regime switching and Bayesian noise
def particle_filter_dlm(y, time_index, num_particles=2000, lag=5):
    T = len(y)
    rng_key = random.PRNGKey(0)

    # Priors for noise variances (log-normal)
    log_sigma_obs_prior_mean = jnp.log(jnp.clip(jnp.nanstd(y) / 10, 0.01, 0.5))
    log_sigma_state_prior_mean = jnp.log(jnp.clip(jnp.nanstd(y) / 50, 0.001, 0.2))

    # Transition probabilities for regime switching
    p_stay_calm = 0.97
    p_stay_crisis = 0.95

    # Initialize particle arrays
    particles_level = jnp.zeros((num_particles, T))
    particles_trend = jnp.zeros((num_particles, T))
    particles_regime = jnp.zeros((num_particles, T), dtype=jnp.int32)
    particles_log_sigma_obs = jnp.ones((num_particles, T)) * log_sigma_obs_prior_mean
    particles_log_sigma_state = jnp.ones((num_particles, T)) * log_sigma_state_prior_mean

    weights = jnp.ones(num_particles) / num_particles

    # Initialize first time step particles
    rng_key, sk1, sk2 = random.split(rng_key, 3)
    particles_level = particles_level.at[:, 0].set(
        y[0] + random.normal(sk1, (num_particles,)) * jnp.exp(log_sigma_obs_prior_mean))
    particles_trend = particles_trend.at[:, 0].set(random.normal(sk2, (num_particles,)) * 0.1)
    particles_regime = particles_regime.at[:, 0].set(0)  # calm regime
    particles_log_sigma_obs = particles_log_sigma_obs.at[:, 0].set(log_sigma_obs_prior_mean)
    particles_log_sigma_state = particles_log_sigma_state.at[:, 0].set(log_sigma_state_prior_mean)

    filtered_level = jnp.zeros(T)
    filtered_trend = jnp.zeros(T)
    filtered_regime_prob = jnp.zeros(T)  # Probability of crisis regime

    ancestor_indices = jnp.zeros((num_particles, T), dtype=jnp.int32)
    jitter_std = 0.02

    # Transition matrix for regimes
    transition_matrix = jnp.array([[p_stay_calm, 1 - p_stay_calm],
                                   [1 - p_stay_crisis, p_stay_crisis]])

    for t in tqdm(range(1, T), desc="Particle Filtering"):
        # Effective sample size (ESS)
        ess = 1. / jnp.sum(weights ** 2)
        if ess < num_particles / 2:
            rng_key, subkey = random.split(rng_key)
            idx = stratified_resample(weights, subkey)

            # Resample particles
            particles_level = particles_level[idx]
            particles_trend = particles_trend[idx]
            particles_regime = particles_regime[idx]
            particles_log_sigma_obs = particles_log_sigma_obs[idx]
            particles_log_sigma_state = particles_log_sigma_state[idx]

            weights = jnp.ones(num_particles) / num_particles

            # Add jitter to maintain diversity (only at last time step)
            rng_key, jkey = random.split(rng_key)
            particles_level = particles_level.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std)
            rng_key, jkey = random.split(rng_key)
            particles_trend = particles_trend.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std)
            rng_key, jkey = random.split(rng_key)
            particles_log_sigma_obs = particles_log_sigma_obs.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std/5)
            rng_key, jkey = random.split(rng_key)
            particles_log_sigma_state = particles_log_sigma_state.at[:, t-1].add(random.normal(jkey, (num_particles,)) * jitter_std/5)

        # Regime transition probabilities per particle
        regime_prev = particles_regime[:, t - 1]
        probs = transition_matrix[regime_prev]  # shape (num_particles, 2)

        # Vectorized sampling of regimes
        rng_key, subkey = random.split(rng_key)
        rng_keys_regime = random.split(subkey, num_particles)
        sample_one = lambda rng_key, p: dist.Categorical(probs=p).sample(rng_key)
        regime_t = jax.vmap(sample_one)(rng_keys_regime, probs)
        particles_regime = particles_regime.at[:, t].set(regime_t)

        # Update noise parameters as random walks
        rng_key, sk_obs, sk_state = random.split(rng_key, 3)
        log_sigma_obs_t = particles_log_sigma_obs[:, t - 1] + random.normal(sk_obs, (num_particles,)) * 0.05
        log_sigma_state_t = particles_log_sigma_state[:, t - 1] + random.normal(sk_state, (num_particles,)) * 0.05

        # Clip log sigmas for numerical stability
        log_sigma_obs_t = jnp.clip(log_sigma_obs_t, jnp.log(0.005), jnp.log(1.0))
        log_sigma_state_t = jnp.clip(log_sigma_state_t, jnp.log(0.001), jnp.log(0.5))

        particles_log_sigma_obs = particles_log_sigma_obs.at[:, t].set(log_sigma_obs_t)
        particles_log_sigma_state = particles_log_sigma_state.at[:, t].set(log_sigma_state_t)

        sigma_obs_t = jnp.exp(log_sigma_obs_t)
        sigma_state_t = jnp.exp(log_sigma_state_t)

        # Inflate noise in crisis regime
        sigma_obs_t = jnp.where(regime_t == 1, sigma_obs_t * 3.0, sigma_obs_t)
        sigma_state_t = jnp.where(regime_t == 1, sigma_state_t * 3.0, sigma_state_t)

        # State transition update
        rng_key, sk1, sk2 = random.split(rng_key, 3)
        new_trend = 0.85 * jnp.tanh(particles_trend[:, t - 1]) + random.normal(sk1, (num_particles,)) * sigma_state_t
        new_level = particles_level[:, t - 1] + new_trend + random.normal(sk2, (num_particles,)) * sigma_state_t

        particles_trend = particles_trend.at[:, t].set(new_trend)
        particles_level = particles_level.at[:, t].set(new_level)

        # Compute weights based on observation likelihood
        log_weights = dist.Normal(new_level, sigma_obs_t).log_prob(y[t])
        log_weights = jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)
        log_weights -= jnp.max(log_weights)  # stabilize
        weights = jnp.exp(log_weights)
        weights = jnp.clip(weights, 1e-12, jnp.inf)  # avoid zeros
        weights /= jnp.sum(weights)

        ancestor_indices = ancestor_indices.at[:, t].set(jnp.arange(num_particles))  # placeholder for ancestry

        # Weighted means
        filtered_level = filtered_level.at[t].set(jnp.sum(particles_level[:, t] * weights))
        filtered_trend = filtered_trend.at[t].set(jnp.sum(particles_trend[:, t] * weights))
        filtered_regime_prob = filtered_regime_prob.at[t].set(jnp.sum(weights * regime_t))

    # Backward smoothing (basic - can be improved)
    smoothed_level = filtered_level
    smoothed_trend = filtered_trend
    smoothed_regime_prob = filtered_regime_prob

    # Credible intervals from particle percentiles
    beta_lower = jnp.percentile(particles_level, 5, axis=0)
    beta_upper = jnp.percentile(particles_level, 95, axis=0)
    beta_mean = jnp.mean(particles_level, axis=0)

    return smoothed_level, beta_mean, beta_lower, beta_upper, particles_level, weights


#  Main DLM Runner
def run_dlm_numpyro(df, target_col, num_particles=5000, lag=5):
    y_series = df[target_col].dropna()
    y = y_series.to_numpy()
    time_index = y_series.index

    smoothed, beta_mean, beta_lower, beta_upper, particles, weights = particle_filter_dlm(
    y, time_index, num_particles=num_particles, lag=lag)


    residuals = y - smoothed
    burn_in = len(y) // 10
    rmse = float(np.sqrt(np.mean(residuals[burn_in:] ** 2)))
    time_index = y_series.index.to_numpy()
    # Plot
    plt.figure(figsize=(12, 8))

    # Top: observed and estimate
    plt.subplot(2, 1, 1)
    plt.plot(time_index, y, label="Observed", color="blue", linewidth=2)
    plt.plot(time_index, beta_mean, label="SMC Estimate", linestyle="--", color="red", linewidth=2)
    plt.fill_between(time_index, beta_lower, beta_upper, alpha=0.3, color="red", label="90% CI")
    plt.title(f"Dynamic Linear Model - Sequential Monte Carlo\n{target_col}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bottom: residuals
    plt.subplot(2, 1, 2)
    residuals = y - beta_mean
    plt.plot(time_index, residuals, 'g-', alpha=0.7, label="Residuals")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/numpyro_dlm_forecast.png", dpi=300, bbox_inches='tight')
    plt.close()

    # RMSE
    #rmse = float(np.sqrt(np.mean((beta_mean - y) ** 2)))
    print(f"NumPyro SMC DLM RMSE: {rmse:.4f}")
    print(f"Final Effective Sample Size: {1.0 / jnp.sum(weights**2):.2f}")

    return particles, rmse
