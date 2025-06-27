from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from src.utils import train_test_split_time_series

def run_arima(df, target_col, train_start, train_end, test_start, test_end, max_p=3, max_d=2, max_q=3):
    warnings.filterwarnings("ignore")

    # Use utility function to split train/test by datetime index
    _, y_train, _, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )

    train = y_train.dropna()
    test = y_test.dropna()

    # 1. Find best ARIMA order on the training set only
    best_aic = np.inf
    best_order = None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        raise ValueError("No suitable ARIMA model found on training data")

    print(f"Selected ARIMA order: {best_order} with AIC: {best_aic:.2f}")

    # 2. Rolling one-step ahead forecast on test set
    history = train.copy()
    forecasts = []
    forecast_lower = []
    forecast_upper = []

    for t in range(len(test)):
        try:
            model = ARIMA(history, order=best_order)
            model_fit = model.fit()
            # Get forecast mean and confidence interval (90%)
            fcast_result = model_fit.get_forecast(steps=1)
            mean_forecast = fcast_result.predicted_mean.iloc[0]
            conf_int = fcast_result.conf_int(alpha=0.1).iloc[0]  # lower, upper
            
            forecasts.append(mean_forecast)
            forecast_lower.append(conf_int[0])
            forecast_upper.append(conf_int[1])
        except Exception:
            forecasts.append(np.nan)
            forecast_lower.append(np.nan)
            forecast_upper.append(np.nan)

        # Append actual observed value to history for next step forecast
        # Use pd.concat instead of deprecated .append()
        history = pd.concat([history, test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    forecast_lower = np.array(forecast_lower)
    forecast_upper = np.array(forecast_upper)

    # 3. Calculate RMSE on valid forecasts only
    valid_idx = ~np.isnan(forecasts)
    rmse = np.sqrt(mean_squared_error(test.values[valid_idx], forecasts[valid_idx]))

    # 4. Plot test period only with styled plot
    plt.figure(figsize=(12, 6))
    plt.plot(test.index.to_numpy(), test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.fill_between(test.index.to_numpy(), forecast_lower, forecast_upper, color="red", alpha=0.3, label="90% CI")
    plt.title(f"ARIMA One-step Ahead Forecast (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/arima_one_step_forecast_test_only.png")
    plt.close()

    # 5. Return forecast series with index aligned to test, RMSE and best order
    forecast_series = pd.Series(forecasts, index=test.index)
    return forecast_series, rmse, best_order
