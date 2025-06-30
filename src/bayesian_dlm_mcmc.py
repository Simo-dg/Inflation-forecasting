import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import train_test_split_time_series
import seaborn as sns

def weighted_percentile(values, weights, percentile):
    """Calculate weighted percentile"""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    cum_weights = cum_weights / cum_weights[-1]  # Normalize to [0, 1]
    
    # Find the percentile
    return np.interp(percentile / 100.0, cum_weights, sorted_values)

def plot_uncertainty_evolution(particles_history, y_test, save_path="results/uncertainty_evolution.png"):
    """Plot how uncertainty evolves over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate uncertainty metrics over time
    uncertainty_std = [np.std(particles) for particles in particles_history]
    uncertainty_iqr = [np.percentile(particles, 75) - np.percentile(particles, 25) for particles in particles_history]
    
    # Plot 1: Uncertainty measures over time
    ax1.plot(y_test.index.to_numpy(), uncertainty_std, label='Standard Deviation', color='red', linewidth=2)
    ax1.plot(y_test.index.to_numpy(), uncertainty_iqr, label='Interquartile Range', color='orange', linewidth=2)
    ax1.set_title('Evolution of Prediction Uncertainty', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Uncertainty Measure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty distribution heatmap
    particles_matrix = np.array(particles_history).T  # Shape: (n_particles, n_timesteps)
    im = ax2.imshow(particles_matrix, aspect='auto', cmap='viridis', alpha=0.8)
    ax2.set_title('Particle Distribution Heatmap Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Particle Index')
    plt.colorbar(im, ax=ax2, label='Particle Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_intervals_analysis(forecast_samples, y_test, save_path="results/prediction_intervals.png"):
    """Analyze prediction interval coverage and reliability"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate various prediction intervals
    intervals = [50, 68, 80, 90, 95, 99]
    coverage_rates = []
    
    for interval in intervals:
        lower_p = (100 - interval) / 2
        upper_p = 100 - lower_p
        
        lower_bounds = np.percentile(forecast_samples, lower_p, axis=1)
        upper_bounds = np.percentile(forecast_samples, upper_p, axis=1)
        
        # Calculate empirical coverage
        in_interval = (y_test.values >= lower_bounds) & (y_test.values <= upper_bounds)
        coverage_rate = np.mean(in_interval) * 100
        coverage_rates.append(coverage_rate)
    
    # Plot 1: Coverage calibration
    ax1.plot(intervals, coverage_rates, 'bo-', linewidth=2, markersize=8)
    ax1.plot([0, 100], [0, 100], 'r--', alpha=0.7, label='Perfect Calibration')
    ax1.set_xlabel('Nominal Coverage (%)')
    ax1.set_ylabel('Empirical Coverage (%)')
    ax1.set_title('Prediction Interval Calibration', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Interval width over time
    width_50 = np.percentile(forecast_samples, 75, axis=1) - np.percentile(forecast_samples, 25, axis=1)
    width_90 = np.percentile(forecast_samples, 95, axis=1) - np.percentile(forecast_samples, 5, axis=1)
    
    ax2.plot(y_test.index.to_numpy(), width_50, label='50% PI Width', color='blue', linewidth=2)
    ax2.plot(y_test.index.to_numpy(), width_90, label='90% PI Width', color='red', linewidth=2)
    ax2.set_title('Prediction Interval Width Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Interval Width')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Probability integral transform (PIT) histogram
    pit_values = []
    for i, obs in enumerate(y_test.values):
        empirical_cdf = np.mean(forecast_samples[i, :] <= obs)
        pit_values.append(empirical_cdf)
    
    ax3.hist(pit_values, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Uniform (ideal)')
    ax3.set_title('Probability Integral Transform', fontweight='bold')
    ax3.set_xlabel('PIT Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Forecast accuracy by uncertainty quantile
    forecast_means = np.mean(forecast_samples, axis=1)
    forecast_std = np.std(forecast_samples, axis=1)
    absolute_errors = np.abs(y_test.values - forecast_means)
    
    # Bin by uncertainty level
    uncertainty_quantiles = pd.qcut(forecast_std, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    error_by_uncertainty = pd.DataFrame({
        'Uncertainty': uncertainty_quantiles,
        'Absolute_Error': absolute_errors
    })
    
    sns.boxplot(data=error_by_uncertainty, x='Uncertainty', y='Absolute_Error', ax=ax4)
    ax4.set_title('Forecast Error by Uncertainty Level', fontweight='bold')
    ax4.set_xlabel('Uncertainty Quantile')
    ax4.set_ylabel('Absolute Error')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_evolution(trace, save_path="results/parameter_evolution.png"):
    """Plot how parameters evolve over time"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract beta samples
    beta_samples = trace.posterior["beta"].values  # Shape: (chains, draws, time)
    sigma_samples = trace.posterior["sigma"].values  # Shape: (chains, draws)
    
    # Plot 1: Beta evolution over time (with uncertainty bands)
    beta_mean = np.mean(beta_samples, axis=(0, 1))
    beta_lower = np.percentile(beta_samples, 5, axis=(0, 1))
    beta_upper = np.percentile(beta_samples, 95, axis=(0, 1))
    
    time_steps = np.arange(len(beta_mean))
    axes[0, 0].plot(time_steps, beta_mean, 'b-', linewidth=2, label='Posterior Mean')
    axes[0, 0].fill_between(time_steps, beta_lower, beta_upper, alpha=0.3, color='blue', label='90% CI')
    axes[0, 0].set_title('State Evolution (β) Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('β Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Beta differences (random walk increments)
    beta_diffs = np.diff(beta_samples, axis=2)
    diff_mean = np.mean(beta_diffs, axis=(0, 1))
    diff_std = np.std(beta_diffs, axis=(0, 1))
    
    axes[0, 1].plot(time_steps[1:], diff_mean, 'g-', linewidth=2, label='Mean Change')
    axes[0, 1].fill_between(time_steps[1:], diff_mean - 2*diff_std, diff_mean + 2*diff_std, 
                        alpha=0.3, color='green', label='±2σ')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('State Changes (Random Walk Increments)', fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Δβ')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sigma posterior distribution
    sigma_flat = sigma_samples.flatten()
    axes[1, 0].hist(sigma_flat, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(sigma_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].axvline(np.median(sigma_flat), color='blue', linestyle='--', linewidth=2, label='Median')
    axes[1, 0].set_title('Observation Noise (σ) Posterior', fontweight='bold')
    axes[1, 0].set_xlabel('σ Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Joint distribution of final state and sigma
    final_beta = beta_samples[:, :, -1].flatten()
    sigma_expanded = np.tile(sigma_samples.flatten(), 1)
    
    # Ensure both arrays have the same length
    min_len = min(len(final_beta), len(sigma_expanded))
    final_beta = final_beta[:min_len]
    sigma_expanded = sigma_expanded[:min_len]
    
    axes[1, 1].scatter(final_beta, sigma_expanded, alpha=0.5, s=10, color='purple')
    axes[1, 1].set_title('Joint Distribution: Final State vs Noise', fontweight='bold')
    axes[1, 1].set_xlabel('Final β Value')
    axes[1, 1].set_ylabel('σ Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_forecast_distribution_evolution(forecast_samples, y_test, save_path="results/forecast_distributions.png"):
    """Plot evolution of forecast distributions over time"""
    n_timesteps = len(y_test)
    
    # Select a subset of timesteps to avoid overcrowding
    timestep_indices = np.linspace(0, n_timesteps-1, min(8, n_timesteps), dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, timestep in enumerate(timestep_indices):
        if i >= len(axes):
            break
            
        # Get forecast samples for this timestep
        samples = forecast_samples[timestep, :]
        observed = y_test.values[timestep]
        
        # Plot histogram
        axes[i].hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for observed value and statistics
        axes[i].axvline(observed, color='red', linewidth=3, label=f'Observed: {observed:.2f}')
        axes[i].axvline(np.mean(samples), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(samples):.2f}')
        axes[i].axvline(np.median(samples), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(samples):.2f}')
        
        # Add prediction intervals
        p5, p95 = np.percentile(samples, [5, 95])
        axes[i].axvspan(p5, p95, alpha=0.2, color='yellow', label='90% PI')
        
        axes[i].set_title(f'Forecast Distribution\nTimestep {timestep+1}', fontweight='bold')
        axes[i].set_xlabel('Forecast Value')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_diagnostics(trace, save_path="results/model_diagnostics.png"):
    """Comprehensive model diagnostics"""
    fig = plt.figure(figsize=(20, 15))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Extract samples
    beta_samples = trace.posterior["beta"].values
    sigma_samples = trace.posterior["sigma"].values
    
    # Plot 1: R-hat values for beta parameters
    ax1 = fig.add_subplot(gs[0, 0])
    rhat_beta = az.rhat(trace.posterior["beta"]).to_array().values.flatten()
    ax1.plot(rhat_beta, 'bo-', markersize=4)
    ax1.axhline(y=1.01, color='red', linestyle='--', label='Good threshold')
    ax1.axhline(y=1.1, color='orange', linestyle='--', label='Acceptable threshold')
    ax1.set_title('R-hat for β Parameters', fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('R-hat')
    ax1.legend()
    
    # Plot 2: Effective sample size
    ax2 = fig.add_subplot(gs[0, 1])
    ess_beta = az.ess(trace, var_names=["beta"]).to_array().values.flatten()
    ax2.plot(ess_beta, 'go-', markersize=4)
    ax2.axhline(y=400, color='red', linestyle='--', label='Minimum recommended')
    ax2.set_title('Effective Sample Size for β', fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('ESS')
    ax2.legend()
    
    # Plot 3: Energy plot
    ax3 = fig.add_subplot(gs[0, 2])
    energy = trace.sample_stats.energy.values.flatten()
    ax3.hist(energy, bins=30, alpha=0.7, density=True, color='purple')
    ax3.set_title('Energy Distribution', fontweight='bold')
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Density')
    
    # Plot 4: Divergences
    ax4 = fig.add_subplot(gs[0, 3])
    divergent = trace.sample_stats.diverging.values.flatten()
    div_rate = np.mean(divergent) * 100
    ax4.bar(['Non-Divergent', 'Divergent'], [100-div_rate, div_rate], 
            color=['green', 'red'], alpha=0.7)
    ax4.set_title(f'Divergent Transitions ({div_rate:.1f}%)', fontweight='bold')
    ax4.set_ylabel('Percentage')
    
    # Plot 5-8: Trace plots for selected beta parameters
    n_params = beta_samples.shape[2]
    selected_params = np.linspace(0, n_params-1, 4).astype(int)
    
    for i, param_idx in enumerate(selected_params):
        ax = fig.add_subplot(gs[1, i])
        for chain in range(beta_samples.shape[0]):
            ax.plot(beta_samples[chain, :, param_idx], alpha=0.7, linewidth=1)
        ax.set_title(f'β[{param_idx}] Trace', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
    
    # Plot 9: Autocorrelation for sigma
    ax9 = fig.add_subplot(gs[2, 0])
    sigma_flat = sigma_samples.flatten()
    autocorr = np.correlate(sigma_flat - np.mean(sigma_flat), 
                        sigma_flat - np.mean(sigma_flat), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]
    lags = np.arange(len(autocorr[:100]))
    ax9.plot(lags, autocorr[:100])
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_title('Autocorrelation (σ)', fontweight='bold')
    ax9.set_xlabel('Lag')
    ax9.set_ylabel('Autocorrelation')
    
    # Plot 10: Posterior predictive check
    ax10 = fig.add_subplot(gs[2, 1:3])
    # This would require posterior predictive samples - simplified version
    ax10.text(0.5, 0.5, 'Posterior Predictive\nCheck Plot\n(Requires posterior\npredictive samples)', 
            ha='center', va='center', transform=ax10.transAxes, fontsize=12)
    ax10.set_title('Posterior Predictive Check', fontweight='bold')
    
    # Plot 11: Summary statistics - FIXED VERSION
    ax11 = fig.add_subplot(gs[2, 3])
    try:
        summary_stats = az.summary(trace, var_names=["sigma"])
        
        # More robust extraction of values
        if isinstance(summary_stats, pd.DataFrame):
            # Method 1: Direct numpy conversion
            sigma_mean = summary_stats['mean'].values[0]
            sigma_std = summary_stats['sd'].values[0]
        else:
            # Fallback: calculate directly from samples
            sigma_mean = np.mean(sigma_samples)
            sigma_std = np.std(sigma_samples)
        
        ax11.axis('off')
        ax11.text(0.1, 0.8, 'Model Summary:', fontweight='bold', transform=ax11.transAxes)
        ax11.text(0.1, 0.6, f'σ mean: {sigma_mean:.4f}', transform=ax11.transAxes)
        ax11.text(0.1, 0.4, f'σ std: {sigma_std:.4f}', transform=ax11.transAxes)
        ax11.text(0.1, 0.2, f'Total parameters: {n_params + 1}', transform=ax11.transAxes)
    except Exception as e:
        # Fallback if summary fails
        ax11.axis('off')
        ax11.text(0.1, 0.8, 'Model Summary:', fontweight='bold', transform=ax11.transAxes)
        ax11.text(0.1, 0.6, f'σ mean: {np.mean(sigma_samples):.4f}', transform=ax11.transAxes)
        ax11.text(0.1, 0.4, f'σ std: {np.std(sigma_samples):.4f}', transform=ax11.transAxes)
        ax11.text(0.1, 0.2, f'Total parameters: {n_params + 1}', transform=ax11.transAxes)
        ax11.text(0.1, 0.0, f'Note: Direct calculation (Error: {str(e)[:30]})', transform=ax11.transAxes, fontsize=8)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_dlm_pymc(df, target_col, train_start, train_end, test_start, test_end, num_particles=1000):
    # Split data
    _, y_train, _, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )

    # Fill missing in training target (if any)
    y = y_train.copy().ffill().bfill()
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
    beta_samples = trace.posterior["beta"].stack(sample=("chain", "draw")).values
    sigma_samples = trace.posterior["sigma"].stack(sample=("chain", "draw")).values

    n_samples = beta_samples.shape[1]

    # Initialize particles at last training time as posterior samples of beta[T_train-1]
    particles = beta_samples[-1, :].copy()
    sigma_particles = sigma_samples.copy()

    # Resample if needed
    if n_samples > num_particles:
        idx = np.random.choice(n_samples, num_particles, replace=False)
        particles = particles[idx]
        sigma_particles = sigma_particles[idx]
        n_samples = num_particles
    else:
        num_particles = n_samples

    forecast_means = []
    forecast_lower = []
    forecast_upper = []
    
    # Store additional data for new plots
    particles_history = []
    forecast_samples_list = []

    weights = np.ones(n_samples) / n_samples
    np.random.seed(420)
    
    # One-step ahead forecasting loop over test points
    for t in range(T_test):
        # Propagate particles by one step of Gaussian RW
        propagated_particles = particles + np.random.normal(0, sigma_particles)
        
        # Store particles for uncertainty analysis
        particles_history.append(propagated_particles.copy())
        forecast_samples_list.append(propagated_particles.copy())

        # Forecast (predict) mean and CI using propagated particles
        mean_forecast = np.mean(propagated_particles)
        lower_forecast = np.percentile(propagated_particles, 5)
        upper_forecast = np.percentile(propagated_particles, 95)

        forecast_means.append(mean_forecast)
        forecast_lower.append(lower_forecast)
        forecast_upper.append(upper_forecast)

        obs = y_test.iloc[t] if hasattr(y_test, 'iloc') else y_test[t]
        log_weights = -0.5 * np.log(2 * np.pi * sigma_particles ** 2) - 0.5 * ((obs - propagated_particles) ** 2) / (sigma_particles ** 2)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Resample particles if effective sample size is low
        ess = 1. / np.sum(weights ** 2)
        if ess < n_samples / 2:
            indices = np.random.choice(n_samples, n_samples, p=weights)
            propagated_particles = propagated_particles[indices]
            sigma_particles = sigma_particles[indices]
            weights = np.ones(n_samples) / n_samples

        particles = propagated_particles
    

    forecast_means = np.array(forecast_means)
    forecast_lower = np.array(forecast_lower)
    forecast_upper = np.array(forecast_upper)
    forecast_samples = np.array(forecast_samples_list)

    rmse_test = np.sqrt(np.mean((forecast_means - y_test.values) ** 2))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecast_means, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.fill_between(y_test.index.to_numpy(), forecast_lower, forecast_upper, color="red", alpha=0.3, label="90% CI")
    plt.title(f"DLM MCMC (RMSE={rmse_test:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/MCMC_forecast.png")
    plt.close()

    #residuals plot
    residuals = y_test.values - forecast_means
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), residuals, label="Residuals", color="green", linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Residuals of DLM MCMC Forecast")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/MCMC_residuals.png")
    plt.close()

    #trace plot
    az.plot_trace(trace, var_names=["beta", "sigma"])
    plt.savefig("results/MCMC_trace.png")     
    plt.close()

    plot_uncertainty_evolution(particles_history, y_test)
    
    # 2. Prediction intervals analysis
    plot_prediction_intervals_analysis(forecast_samples, y_test)
    
    # 3. Parameter evolution
    plot_parameter_evolution(trace)
    
    # 4. Forecast distribution evolution
    plot_forecast_distribution_evolution(forecast_samples, y_test)
    
    # 5. Comprehensive model diagnostics
    plot_model_diagnostics(trace)


    """ #aic, bic, waic, loo
    log_likelihood = np.asarray(trace.log_likelihood["y_obs"])
    ll_samples = log_likelihood.sum(axis=0).flatten()
    mean_ll = np.mean(ll_samples)
    n_params = T_train + 1  # Number of parameters (beta + sigma)
    n_obs = T_train
    aic = -2 * mean_ll + 2 * n_params
    bic = -2 * mean_ll + np.log(n_obs) * n_params
    waic_result = az.waic(trace, pointwise=True)
    loo_result = az.loo(trace, pointwise=True)
    waic = waic_result.elpd_waic * -2  # Convert to deviance scale
    loo = loo_result.elpd_loo * -2    # Convert to deviance scale
    p_loo = loo_result.p_loo

    # Save diagnostics
    diagnostics = {
        "AIC": aic,
        "BIC": bic,
        "WAIC": waic,
        "LOO": loo,
        "LOO_p_eff": p_loo
    }

    # Save Report as CSV
    results_df = pd.DataFrame([{
        "RMSE_Test": rmse_test,
        "AIC": diagnostics["AIC"],
        "BIC": diagnostics["BIC"],
        "WAIC": diagnostics["WAIC"],
        "LOO_CV": diagnostics["LOO"],
        "LOO_p_eff": diagnostics["LOO_p_eff"]
    }])
    results_df.to_csv("results/model_performance_metrics.csv", index=False) """
    return trace, rmse_test
