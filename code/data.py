import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transform import transform_mobile_data 
def read_preprocessed_data(
    file_name="train.csv",
    outliers=True,
    scaling_method=None,              # 'standard' lub 'minmax'
    log_transform_cols=None,
    onehot=False
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    file_path = os.path.join(root_dir, "data", file_name)
    print(f"Wczytuję dane z: {file_path}")

    df = pd.read_csv(file_path)
    df = df.dropna()

    if outliers:
        df = remove_outliers(df)

    y = df["price_range"]
    X = df.drop("price_range", axis=1)

    # One-hot encoding cech jakościowych
    # if onehot:
    #     qualitative = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
    #     encoder = OneHotEncoder(sparse=False, drop='if_binary')
    #     encoded = encoder.fit_transform(X[qualitative])
    #     encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(qualitative), index=X.index)
    #     X = pd.concat([X.drop(columns=qualitative), encoded_df], axis=1)

    # Skalowanie i logarytmowanie
    if scaling_method:
        X = transform_mobile_data(X, scaling_method=scaling_method, log_transform_cols=log_transform_cols)

    return X, y

def outliers_statistics(df):
    outliers_count = []

    for col in df.columns:
        if df[col].dtypes in [float, int]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * IQR)) | (df[col] > (q3 + 1.5 * IQR))]
            outliers_count.append(outliers.shape[0])
        else:
            outliers_count.append(0)

    outliers_percentage = [count / len(df) for count in outliers_count]

    outliers_df = pd.DataFrame(
        [outliers_count, outliers_percentage], columns=df.columns
    )
    outliers_df.index = ["Ilość wartości skrajnych", "Procent wartości skrajnych"]

    return outliers_df

def remove_outliers(df2):
    outliers_percentages = outliers_statistics(df2)
    df = df2.copy()
    for i, col in enumerate(df.columns):
        if df[col].dtypes in [float, int]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR
            if outliers_percentages.iloc[1, i] < 0.05:
                df[col] = df[col].where(~((df[col] < lower_bound) | (df[col] > upper_bound)))
            else:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df.dropna()

def general_statistics(df):
    stat_cols = [col for col in df.columns if df[col].nunique() > 7]
    num_cols = df[stat_cols]
    statystyki = {
        "Średnia": num_cols.mean(),
        "Mediana": num_cols.median(),
        "Minimum": num_cols.min(),
        "Maksimum": num_cols.max(),
        "Odchylenie Standardowe": num_cols.std(),
        "Skośność": num_cols.skew(),
    }
    return pd.DataFrame(statystyki)

def dataframe_to_latex_table(df, row_label='Zmienna'):
    latex_str = []
    latex_str.append(r'\begin{adjustbox}{valign=c, width=\textwidth}')
    latex_str.append(r'\begin{tabular}{|' + 'r|' * (len(df.columns)+1) + '}')
    latex_str.append(r'\hline')
    headers = ['\\textbf{' + row_label + '}'] + ['\\textbf{' + str(col) + '}' for col in df.columns]
    latex_str.append(' & '.join(headers) + r' \\')
    latex_str.append(r'\hline')
    for index, row in df.iterrows():
        row_items = [str(index).replace('_', r'\_')] + [f"{val:.2f}" if isinstance(val, (float, int)) else str(val) for val in row]
        latex_str.append(' & '.join(row_items) + r' \\')
    latex_str.append(r'\hline')
    latex_str.append(r'\end{tabular}')
    latex_str.append(r'\end{adjustbox}')
    return '\n'.join(latex_str)
