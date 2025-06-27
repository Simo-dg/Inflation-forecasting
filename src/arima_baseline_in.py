from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd

def run_arima_in(df, target_col, min_train_size=150, max_p=3, max_d=2, max_q=3):
    warnings.filterwarnings("ignore")

    y = df[target_col].dropna()

    # 1. Find best ARIMA order on initial training set only
    train_init = y.iloc[:min_train_size]
    best_aic = np.inf
    best_order = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train_init, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        raise ValueError("No suitable ARIMA model found on initial training data")

    print(f"Selected ARIMA order: {best_order} with AIC: {best_aic:.2f}")

    forecasts = []
    actuals = []
    forecast_index = []

    # 2. Rolling forecast with fixed best_order
    for i in range(min_train_size, len(y)):
        train = y.iloc[:i]
        try:
            model = ARIMA(train, order=best_order)
            model_fit = model.fit()
            fcast = model_fit.forecast(steps=1)
            forecasts.append(fcast.iloc[0])
        except Exception:
            forecasts.append(np.nan)

        actuals.append(y.iloc[i])
        forecast_index.append(y.index[i])

    # 3. Create series for forecasts and actuals
    forecast_series = pd.Series(forecasts, index=forecast_index)
    actual_series = pd.Series(actuals, index=forecast_index)

    # 4. Calculate RMSE on non-NaN forecasts
    valid_idx = forecast_series.notna()
    rmse = np.sqrt(mean_squared_error(actual_series[valid_idx], forecast_series[valid_idx]))

    # 5. Plot actual vs forecast
    plt.figure(figsize=(12,5))
    plt.plot(y, label='Actual')
    plt.plot(forecast_series, label=f'One-step ARIMA forecast (order={best_order})', linestyle='--')
    plt.title('ARIMA')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/arima_forecast_in.png")
    plt.close()

    return forecast_series, rmse, best_order
