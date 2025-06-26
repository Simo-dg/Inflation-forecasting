def train_test_split_series(series, test_size):
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test
