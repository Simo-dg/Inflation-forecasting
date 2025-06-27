import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from src.utils import train_test_split_time_series

def run_ols(df, target_col, train_start, train_end, test_start, test_end):
    # Split data
    X_train, y_train, X_test, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )

    # Align train X,y
    X_train, y_train = X_train.dropna(), y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    # We'll do rolling 1-step ahead forecasts on test
    history_X = X_train.copy()
    history_y = y_train.copy()
    forecasts = []

    for i in range(len(X_test)):
        # Fit model on current history
        model = LinearRegression()
        model.fit(history_X, history_y)

        # Predict 1-step ahead (next test sample)
        x_next = X_test.iloc[i:i+1]
        fcast = model.predict(x_next)[0]
        forecasts.append(fcast)

        # Add observed test data to history for next iteration
        history_X = pd.concat([history_X, x_next])
        history_y = pd.concat([history_y, y_test.iloc[i:i+1]])

    forecasts = np.array(forecasts)

    # Compute RMSE on test set
    rmse = np.sqrt(mean_squared_error(y_test.values, forecasts))

    # Plot test period only with your preferred style
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"OLS (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ols.png")
    plt.close()

    return model, rmse


def run_ols_feature_selection(df, target_col, train_start, train_end, test_start, test_end, n_features=5):
    # Split data
    X_train, y_train, X_test, y_test = train_test_split_time_series(
        df, train_start, train_end, test_start, test_end, target_col
    )

    X_train, y_train = X_train.dropna(), y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    history_X = X_train.copy()
    history_y = y_train.copy()
    forecasts = []
    selected_features = None

    for i in range(len(X_test)):
        # Fit RFE selector + model on current history
        base_model = LinearRegression()
        selector = RFE(base_model, n_features_to_select=n_features)
        selector.fit(history_X, history_y)
        selected_features = history_X.columns[selector.support_]

        model = LinearRegression()
        model.fit(history_X[selected_features], history_y)

        x_next = X_test.iloc[i:i+1][selected_features]
        fcast = model.predict(x_next)[0]
        forecasts.append(fcast)

        # Update history with observed test data
        history_X = pd.concat([history_X, X_test.iloc[i:i+1]])
        history_y = pd.concat([history_y, y_test.iloc[i:i+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test.values, forecasts))

    # Plot test period only
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"OLS with Feature Selection (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ols_rfe.png")
    plt.close()

    return model, selected_features, rmse
