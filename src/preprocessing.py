import pandas as pd
import numpy as np
def load_and_clean_data(path):
    df = pd.read_csv(path)

    columns_to_drop = [
        'GDP', 'HICP - YoY', 'CPI - index', 'HICP - index', 
        'PPI - index', 'PPI - YoY', 'Loans_upto_1M', 'Loans_house',
        'Loans_over_1M','Retain_trade_index','wage rate - index', 'wage rate - YoY'
    ]
    
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Fix TIME_PERIOD strings and convert to datetime
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y-M%m')
    df = df.loc[df['TIME_PERIOD'] >= '2001-12-31']
    df = df.loc[df['TIME_PERIOD'] <= '2024-01-01']

    df.set_index('TIME_PERIOD', inplace=True)
    print(df.head())

    # Replace '.' with NaN before numeric conversion
    df = df.replace('.', np.nan)

    # Strip whitespace from strings before conversion
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Clean percentage signs before conversion
    if 'AEX change' in df.columns:
        df['AEX change'] = df['AEX change'].astype(str).str.replace('%', '', regex=False)

    # Convert specific columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Interpolate missing values using time index
    df = df.interpolate(method='time')

    return df
