import jax
import jax.numpy as jnp
import jax.random as random
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from scipy import stats
from tqdm import tqdm
from src.utils import train_test_split_time_series
import seaborn as sns

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

    return filtered_level, beta_mean, beta_lower, beta_upper, particles_level, particles_trend, particles_regime, particles_log_sigma_obs[:, -1], particles_log_sigma_state[:, -1], weights, rng_key, filtered_regime_prob


def plot_regime_switching_analysis(time_index, y_data, filtered_regime_prob, particles_regime, target_col):
    """Create sophisticated regime switching visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main time series with regime coloring
    ax1 = fig.add_subplot(gs[0, :])
    
    # Convert JAX array to numpy for indexing
    filtered_regime_prob_np = np.array(filtered_regime_prob)
    crisis_periods = filtered_regime_prob_np > 0.5
    time_index = np.array(time_index)  
    
    # Add shaded regions for crisis periods
    in_crisis = False
    start_crisis = None
    
    for i, is_crisis in enumerate(crisis_periods):
        if is_crisis and not in_crisis:
            start_crisis = time_index[i]
            in_crisis = True
        elif not is_crisis and in_crisis:
            ax1.axvspan(start_crisis, time_index[i], alpha=0.2, color='red', 
                       label='Crisis Regime' if start_crisis == time_index[np.where(crisis_periods)[0][0]] else "")
            in_crisis = False
    
    # Handle case where series ends in crisis
    if in_crisis:
        ax1.axvspan(start_crisis, time_index[-1], alpha=0.2, color='red', label='Crisis Regime')
    
    ax1.plot(time_index, y_data, 'k-', linewidth=1.5, alpha=0.8, label=f'{target_col}')
    ax1.set_title('Time Series with Regime Classification', fontsize=14, fontweight='bold')
    ax1.set_ylabel(target_col, fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Regime probability evolution
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(time_index, 0, filtered_regime_prob_np, alpha=0.6, color='crimson', label='Crisis Probability')
    ax2.fill_between(time_index, filtered_regime_prob_np, 1, alpha=0.6, color='steelblue', label='Normal Probability')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
    ax2.set_title('Regime Probability Evolution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('P(Crisis Regime)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Particle regime distribution heatmap (sample recent periods)
    ax3 = fig.add_subplot(gs[2, :])
    n_recent = min(200, len(time_index))
    recent_particles = np.array(particles_regime[:, -n_recent:])  # Convert to numpy
    
    # Create heatmap data
    heatmap_data = []
    for t in range(recent_particles.shape[1]):
        regime_counts = np.bincount(recent_particles[:, t], minlength=2)
        heatmap_data.append(regime_counts / np.sum(regime_counts))
    
    heatmap_data = np.array(heatmap_data).T
    
    im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_title(f'Particle Regime Distribution (Last {n_recent} periods)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Regime State')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'Crisis'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Particle Fraction', rotation=270, labelpad=15)
    
    # Regime transition statistics
    ax4 = fig.add_subplot(gs[3, 0])
    regime_changes = np.diff(filtered_regime_prob_np > 0.5).astype(int)
    normal_to_crisis = np.sum(regime_changes == 1)
    crisis_to_normal = np.sum(regime_changes == -1)
    
    categories = ['Normal→Crisis', 'Crisis→Normal']
    counts = [normal_to_crisis, crisis_to_normal]
    bars = ax4.bar(categories, counts, color=['darkred', 'darkblue'], alpha=0.7)
    ax4.set_title('Regime Transitions', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(count)), ha='center', va='bottom', fontweight='bold')
    
    # Regime duration analysis
    ax5 = fig.add_subplot(gs[3, 1])
    
    # Calculate regime durations
    regime_binary = (filtered_regime_prob_np > 0.5).astype(int)
    durations = {'Normal': [], 'Crisis': []}
    
    current_regime = regime_binary[0]
    duration = 1
    
    for i in range(1, len(regime_binary)):
        if regime_binary[i] == current_regime:
            duration += 1
        else:
            regime_name = 'Crisis' if current_regime == 1 else 'Normal'
            durations[regime_name].append(duration)
            current_regime = regime_binary[i]
            duration = 1
    
    # Add final duration
    regime_name = 'Crisis' if current_regime == 1 else 'Normal'
    durations[regime_name].append(duration)
    
    # Create violin plot
    all_durations = []
    labels = []
    for regime, durs in durations.items():
        if durs:  # Only add if there are durations
            all_durations.extend(durs)
            labels.extend([regime] * len(durs))
    
    if all_durations:
        df_durations = pd.DataFrame({'Duration': all_durations, 'Regime': labels})
        sns.violinplot(data=df_durations, x='Regime', y='Duration', ax=ax5, palette=['steelblue', 'crimson'])
        ax5.set_title('Regime Duration Distribution', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Duration (periods)')
    
    plt.tight_layout()
    plt.savefig('results/regime_switching_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bayesian_uncertainty_analysis(time_index, y_data, particles_level, particles_trend, 
                                     particles_log_sigma_obs, particles_log_sigma_state, target_col):
    """Create comprehensive Bayesian uncertainty visualization"""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    time_index = np.array(time_index, copy=True)
    y_data = np.array(y_data, copy=True)
    particles_level = np.array(particles_level, copy=True)
    particles_trend = np.array(particles_trend, copy=True)
    particles_log_sigma_obs = np.array(particles_log_sigma_obs, copy=True)
    particles_log_sigma_state = np.array(particles_log_sigma_state, copy=True)


    
    
    # Particle evolution fan chart
    ax1 = fig.add_subplot(gs[0, :])
    
    # Sample subset of particles for visualization
    n_particles_plot = min(100, particles_level.shape[0])
    indices = np.random.choice(particles_level.shape[0], n_particles_plot, replace=False)
    
    for i in indices[:50]:  # Plot first 50 for clarity
        ax1.plot(time_index, particles_level[i], alpha=0.1, color='blue', linewidth=0.5)
    
    # Add percentiles
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'orange', 'red']
    alphas = [0.8, 0.6, 1.0, 0.6, 0.8]
    
    for p, color, alpha in zip(percentiles, colors, alphas):
        level_p = np.percentile(particles_level, p, axis=0)
        ax1.plot(time_index, level_p, color=color, alpha=alpha, linewidth=2, 
                label=f'{p}th percentile' if p in [5, 50, 95] else None)
    
    # Fill between extreme percentiles
    level_5 = np.percentile(particles_level, 5, axis=0)
    level_95 = np.percentile(particles_level, 95, axis=0)
    ax1.fill_between(time_index, level_5, level_95, alpha=0.2, color='gray', label='90% Credible Interval')
    
    ax1.plot(time_index, y_data, 'ko', markersize=2, alpha=0.7, label='Observed')
    ax1.set_title('Particle Filter: Posterior Evolution with Uncertainty Quantification', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'{target_col}', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Posterior distribution at different time points
    ax2 = fig.add_subplot(gs[1, 0])
    n_times = len(time_index)
    time_points = [n_times//4, n_times//2, 3*n_times//4, -1]
    colors_hist = ['blue', 'green', 'orange', 'red']
    
    for i, (tp, color) in enumerate(zip(time_points, colors_hist)):
        weights = np.ones(len(particles_level[:, tp])) / len(particles_level[:, tp])  # Uniform for simplicity
        # Convert to string format for the label
        time_label = time_index[tp].strftime("%Y-%m") if hasattr(time_index[tp], 'strftime') else str(time_index[tp])[:7]
        ax2.hist(particles_level[:, tp], bins=30, alpha=0.6, color=color, density=True,
                label=f't={time_label}')
    
    ax2.set_title('Posterior Distributions at Key Time Points', fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'Level ({target_col})')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Trend evolution
    ax3 = fig.add_subplot(gs[1, 1])
    trend_mean = np.mean(particles_trend, axis=0)
    trend_std = np.std(particles_trend, axis=0)
    
    ax3.plot(time_index, trend_mean, 'g-', linewidth=2, label='Mean Trend')
    ax3.fill_between(time_index, trend_mean - 2*trend_std, trend_mean + 2*trend_std, 
                    alpha=0.3, color='green', label='±2σ Interval')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Trend Component Evolution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Trend')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Parameter evolution (log-scale variances)
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Get parameter evolution (using last values as proxy)
    sigma_obs_samples = np.exp(particles_log_sigma_obs)
    sigma_state_samples = np.exp(particles_log_sigma_state)
    
    ax4.hist(sigma_obs_samples, bins=50, alpha=0.7, color='blue', density=True, label='σ_obs')
    ax4.hist(sigma_state_samples, bins=50, alpha=0.7, color='red', density=True, label='σ_state')
    ax4.set_title('Final Parameter Posterior Distributions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Standard Deviation')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Uncertainty quantification over time
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate various uncertainty measures
    credible_width = np.percentile(particles_level, 95, axis=0) - np.percentile(particles_level, 5, axis=0)
    posterior_variance = np.var(particles_level, axis=0)
    
    ax5_twin = ax5.twinx()
    
    l1 = ax5.plot(time_index, credible_width, 'b-', linewidth=2, label='90% Credible Width')
    l2 = ax5_twin.plot(time_index, posterior_variance, 'r-', linewidth=2, label='Posterior Variance')
    
    ax5.set_title('Uncertainty Measures Over Time', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Credible Interval Width', color='b')
    ax5_twin.set_ylabel('Posterior Variance', color='r')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Predictive intervals analysis
    ax6 = fig.add_subplot(gs[3, :])
    
    # Calculate prediction accuracy within credible intervals
    level_5 = np.percentile(particles_level, 5, axis=0)
    level_25 = np.percentile(particles_level, 25, axis=0)
    level_75 = np.percentile(particles_level, 75, axis=0)
    level_95 = np.percentile(particles_level, 95, axis=0)
    
    # Coverage indicators
    in_50 = (y_data >= level_25) & (y_data <= level_75)
    in_90 = (y_data >= level_5) & (y_data <= level_95)
    
    ax6.plot(time_index, y_data, 'ko', markersize=3, alpha=0.8, label='Observed')
    ax6.fill_between(time_index, level_5, level_95, alpha=0.2, color='red', label='90% Credible')
    ax6.fill_between(time_index, level_25, level_75, alpha=0.4, color='blue', label='50% Credible')
    
    # Highlight points outside intervals
    outside_90 = ~in_90
    if np.any(outside_90):
        ax6.scatter(time_index[outside_90], y_data[outside_90], 
                   color='red', s=30, marker='x', label='Outside 90% CI', zorder=5)
    
    coverage_50 = np.mean(in_50) * 100
    coverage_90 = np.mean(in_90) * 100
    
    ax6.set_title(f'Credible Interval Coverage (50%: {coverage_50:.1f}%, 90%: {coverage_90:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax6.set_ylabel(f'{target_col}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/bayesian_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()




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
    filtered_level, beta_mean, beta_lower, beta_upper, particles_level, particles_trend, particles_regime, last_log_sigma_obs, last_log_sigma_state, weights, rng_key, filtered_regime_prob = particle_filter_dlm(
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
    plt.title(f"DLM - SMC (RMSE={rmse_test:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.savefig("results/DLM_SMC.png")
    plt.close()

    plot_regime_switching_analysis(train_time_index, y_train_np, filtered_regime_prob, 
                                 particles_regime, target_col)

    plot_bayesian_uncertainty_analysis(train_time_index, y_train_np, particles_level, 
                                     particles_trend, last_log_sigma_obs, 
                                     last_log_sigma_state, target_col)

    return forecast_level, rmse_test

