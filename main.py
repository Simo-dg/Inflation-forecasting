import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.preprocessing import load_and_clean_data
from src.arima_baseline import run_arima
from src.regression_ols import run_ols, run_ols_feature_selection
from src.ml_models import run_rf, run_xgboost, run_mlp
from src.bayesian_dlm_mcmc import run_dlm_pymc
from src.bayesian_dlm_smc import run_dlm_numpyro
from src.metrics import plot_model_comparison
from src.bayesian_dlm_reg import run_dlm_dynamic_regression
from src.regression_ols_in import run_ols_in, run_ols_feature_selection_in
from src.ml_models_in import run_rf_in, run_xgboost_in, run_mlp_in
from src.bayesian_dlm_mcmc_in import run_dlm_pymc_in
from src.bayesian_dlm_smc_in import run_dlm_numpyro_in
from src.bayesian_dlm_reg_in import run_dlm_dynamic_regression_in
from src.arima_baseline_in import run_arima_in

# 📁 Create folders if they don't exist
os.makedirs("results", exist_ok=True)

# ⚙️ Dataset and variables
data_path = "dataset.csv"
target_col = "CPI - YoY"

print("🚀 Starting Inflation Forecasting Model Comparison")
print("=" * 60)

# 📊 1. Preprocessing
print("📊 Loading and preprocessing data...")
df = load_and_clean_data(data_path)
print(f"✅ Data loaded: {len(df)} observations")
print()

# Define models to run
models = [
    ("ARIMA", "📈 Classical Time Series"),
    ("OLS", "📈 Linear Regression"),
    ("OLS + RFE", "📈 Feature Selection"),
    ("Random Forest", "🤖 Ensemble Learning"),
    ("XGBoost", "🤖 Gradient Boosting"),
    ("MLP", "🤖 Neural Network"),
    ("Bayesian DLM (MCMC)", "🧠 Bayesian MCMC"),
    ("Bayesian DLM (SMC)", "🧠 Sequential Monte Carlo"),
    ("Bayesian DLM (DR)", "🧠 Dynamic Coefficients")
]

# Initialize results dictionary
rmse_results_out = {}

# Create main progress bar
main_pbar = tqdm(models, desc="🔄 Running Models", unit="model", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

train_start="2001-12-31"
train_end="2019-12-31"
test_start="2020-01-01"
test_end="2023-12-31"

for model_name, model_desc in main_pbar:
    main_pbar.set_description(f"🔄 {model_desc}")
    
    try:
        start_time = time.time()
        
        if model_name == "ARIMA":
            forecast, rmse, best_order = run_arima(
                df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

        elif model_name == "OLS":
            model, rmse = run_ols(
                df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

        elif model_name == "OLS + RFE":
            model, selected_features, rmse = run_ols_feature_selection(
                df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_features=5 
            )
            
        elif model_name == "Random Forest":
            _, rmse = run_rf(df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
        elif model_name == "XGBoost":
            _, rmse = run_xgboost(df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
        elif model_name == "MLP":
            _, rmse = run_mlp(df, target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
        elif model_name == "Bayesian DLM (MCMC)":
            _, rmse = run_dlm_pymc(
                df,
                target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            
        elif model_name == "Bayesian DLM (SMC)":
            _, rmse = run_dlm_numpyro(
                df,
                target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

        elif model_name == "Bayesian DLM (DR)":
            trace, rmse = run_dlm_dynamic_regression(
                df,
                target_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

        # Store result
        rmse_results_out[model_name] = rmse
        
        # Calculate execution time
        exec_time = time.time() - start_time
        
        # Update progress bar with result
        main_pbar.set_postfix({
            'RMSE': f'{rmse:.4f}', 
            'Time': f'{exec_time:.1f}s'
        })
        
        print(f"\n✅ {model_name}: RMSE = {rmse:.4f} (took {exec_time:.1f}s)")
        
    except Exception as e:
        print(f"\n❌ Error running {model_name}: {str(e)}")
        rmse_results_out[model_name] = float('inf')  # Mark as failed
        continue

# 📊 2. In-sample models
rmse_results_in = {}

# Create main progress bar
main_pbar = tqdm(models, desc="🔄 Running Models", unit="model", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

for model_name, model_desc in main_pbar:
    main_pbar.set_description(f"🔄 {model_desc}")
    
    try:
        start_time = time.time()
        
        if model_name == "ARIMA":
            forecast, rmse, best_order = run_arima_in(df, target_col)

        elif model_name == "OLS":
            model, rmse = run_ols_in(df, target_col)

        elif model_name == "OLS + RFE":
            model, selected_features, rmse = run_ols_feature_selection_in(df, target_col, n_features=5)

        elif model_name == "Random Forest":
            _, rmse = run_rf_in(df, target_col)
            
        elif model_name == "XGBoost":
            _, rmse = run_xgboost_in(df, target_col)
            
        elif model_name == "MLP":
            _, rmse = run_mlp_in(df, target_col)

        elif model_name == "Bayesian DLM (MCMC)":
            _, rmse = run_dlm_pymc_in(df, target_col)

            
        elif model_name == "Bayesian DLM (SMC)":
            _, rmse = run_dlm_numpyro_in(df, target_col)

        elif model_name == "Bayesian DLM (DR)":
            trace, rmse = run_dlm_dynamic_regression_in(df, target_col)

        # Store result
        rmse_results_in[model_name] = rmse
        
        # Calculate execution time
        exec_time = time.time() - start_time
        
        # Update progress bar with result
        main_pbar.set_postfix({
            'RMSE': f'{rmse:.4f}', 
            'Time': f'{exec_time:.1f}s'
        })
        
        print(f"\n✅ {model_name}: RMSE = {rmse:.4f} (took {exec_time:.1f}s)")
        
    except Exception as e:
        print(f"\n❌ Error running {model_name}: {str(e)}")
        rmse_results_in[model_name] = float('inf')  # Mark as failed
        continue




print("\n" + "=" * 60)
print("📊 Generating comparison plots and saving results...")

# 📊 5. Comparison
plot_model_comparison(rmse_results_out, filename="results/model_comparison_out_sample.png")
plot_model_comparison(rmse_results_in, filename="results/model_comparison_in_sample.png")


# 📝 Save RMSE to CSV
results_df = pd.DataFrame.from_dict(rmse_results_out, orient='index', columns=['RMSE'])
results_df = results_df.sort_values('RMSE')  # Sort by best performance
results_df.to_csv("results/rmse_summary.csv")

results_df_in = pd.DataFrame.from_dict(rmse_results_in, orient='index', columns=['RMSE'])
results_df_in = results_df_in.sort_values('RMSE')  # Sort by best performance
results_df_in.to_csv("results/rmse_summary_in.csv")

print("\n🏆 FINAL RESULTS RANKING:")
print("-" * 40)
for i, (model, rmse) in enumerate(results_df.iterrows(), 1):
    status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
    print(f"{status} {model:<25} RMSE: {rmse['RMSE']:.4f}")

print("\n🏆 FINAL RESULTS RANKING (In-sample):")
print("-" * 40)
for i, (model, rmse) in enumerate(results_df_in.iterrows(), 1):
    status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
    print(f"{status} {model:<25} RMSE: {rmse['RMSE']:.4f}")

print(f"\n✅ All models completed! Results saved in /results")
print("📁 Files generated:")
print("   • rmse_summary.csv - Detailed results")
print("   • model_comparison.png - Visual comparison")
print("   • Individual model plots in /results folder")

# Optional: Show execution summary
total_successful = sum(1 for rmse in rmse_results_out.values() if rmse != float('inf'))
print(f"\n📈 Execution Summary: {total_successful}/{len(models)} models completed successfully")

total_successful_in = sum(1 for rmse in rmse_results_in.values() if rmse != float('inf'))
print(f"📈 In-sample Execution Summary: {total_successful_in}/{len(models)} models completed succesfully")