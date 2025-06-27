def train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col):
    """
    Splits a DataFrame into train and test sets based on datetime index.

    Args:
        df (pd.DataFrame): Full dataset with datetime index.
        train_start (str or pd.Timestamp): Training start date.
        train_end (str or pd.Timestamp): Training end date.
        test_start (str or pd.Timestamp): Test start date.
        test_end (str or pd.Timestamp): Test end date.
        target_col (str): Name of the target column.

    Returns:
        X_train, y_train, X_test, y_test (pd.DataFrame or pd.Series): Split data.
    """
    df_train = df.loc[train_start:train_end].copy()
    df_test = df.loc[test_start:test_end].copy()

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test

