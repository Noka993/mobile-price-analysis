import pandas as pd
import  os
def read_preprocessed_data(file_name= "train.csv",outliers=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    file_path = os.path.join(root_dir, "data", file_name)
    print(f"Wczytuję dane z: {file_path}")

    df = pd.read_csv(file_path)
    # Brakujące dane
    df = df.dropna()
    if(outliers):
        df=remove_outliers(df)

    return df
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

    outliers_percentage = []
    for value in outliers_count:
        outliers_percentage.append(float(value) / len(df))

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
            q1 = df2[col].quantile(0.25)
            q3 = df2[col].quantile(0.75)
            IQR = q3 - q1
            upper_bound = q3 + 1.5 * IQR
            lower_bound = q1 - 1.5 * IQR
            outliers_percentage = outliers_percentages.iloc[1, i]
            if outliers_percentage < 0.05:
                df[col] = df[col].where(
                    ~((df[col] < lower_bound) | (df[col] > upper_bound))
                )
            else:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    cleaned_df = df.dropna()

    return cleaned_df
