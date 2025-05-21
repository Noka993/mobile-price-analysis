import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def transform_mobile_data(
    df: pd.DataFrame, 
    scaling_method: str = 'standard',  # 'standard' lub 'minmax'
    log_transform_cols: list = None
) -> pd.DataFrame:
    """
    Funkcja skaluje numeryczne kolumny i opcjonalnie wykonuje logarytmowanie wybranych kolumn.
    
    Parametry:
    - df: DataFrame z danymi
    - scaling_method: 'standard' dla StandardScaler lub 'minmax' dla MinMaxScaler
    - log_transform_cols: lista kolumn do logarytmowania (np. ['ram', 'battery_power'])
    
    Zwraca:
    - przekształcony DataFrame
    """

    # Kolumny do logarytmowania
    # log_transform_cols = ['battery_power', 'ram', 'int_memory', 'pc', 'fc']

    numeric_cols = [
        'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 
        'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'
    ]
    
    df_transformed = df.copy()
    
    # Logarytmowanie
    if log_transform_cols:
        for col in log_transform_cols:
            if col in df_transformed.columns:
                df_transformed[col] = np.log1p(df_transformed[col])
            else:
                print(f"Uwaga: kolumna '{col}' nie istnieje w DataFrame.")
    
    # Wybór skalera
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaling_method musi być 'standard' lub 'minmax'")
    
    # Skalowanie
    df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    return df_transformed

# --- przykładowe użycie ---

# df_scaled_log = transform_mobile_data(df, scaling_method='minmax', log_transform_cols=['ram', 'battery_power'])
# print(df_scaled_log.head())
