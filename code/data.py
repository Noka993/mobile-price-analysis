import pandas as pd
import  os

from PIL.ImageOps import scale
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def read_preprocessed_data(file_name= "train.csv",outliers=True,std=False,onehot=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    file_path = os.path.join(root_dir, "data", file_name)
    print(f"Wczytuję dane z: {file_path}")

    df = pd.read_csv(file_path)
    # Brakujące dane
    df = df.dropna()
    if(outliers):
        df=remove_outliers(df)
    jakosciowe=["blue","dual_sim","four_g","three_g","touch_screen","wifi"]
    temp=df.drop("price_range",axis=1)
    ilosciowe=[]
    for col in temp.columns:
        if col not in jakosciowe:
            ilosciowe.append(col)
    if (std):
        scaler = StandardScaler()
        temp[ilosciowe]=scaler.fit_transform(temp[ilosciowe])


    return  temp,df.iloc[:, -1]
   # df= df.drop

   # standardscaler
  #  onehotencoder

   # return df
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
def general_statistics(df):
    stat_cols = [col for col in df.columns if df[col].nunique() > 7]
    num_cols = pd.DataFrame(
        df, columns=stat_cols
    )
    statystyki = {
        "Średnia": num_cols.mean(),
        "Mediana": num_cols.median(),
        "Minimum": num_cols.min(),
        "Maksimum": num_cols.max(),
        "Odchylenie Standardowe": num_cols.std(),
        "Skośność": num_cols.skew(),
    }
    statystyki = pd.DataFrame(statystyki)
    return statystyki


def dataframe_to_latex_table(df, row_label='Zmienna'):
    latex_str = []
    latex_str.append(r'\begin{adjustbox}{valign=c, width=\textwidth}')
    latex_str.append(r'\begin{tabular}{|' + 'r|' * (len(df.columns)+1) + '}')
    latex_str.append(r'\hline')
    # Add header row
    headers = ['\\textbf{' + row_label + '}'] + ['\\textbf{' + str(col) + '}' for col in df.columns]
    latex_str.append(' & '.join(headers) + r' \\')
    latex_str.append(r'\hline')
    # Add data rows
    for index, row in df.iterrows():
        row_items = [str(index).replace('_', r'\_')] + [f"{val:.2f}" if isinstance(val, (float, int)) else str(val) for val in row]
        latex_str.append(' & '.join(row_items) + r' \\')
    latex_str.append(r'\hline')
    latex_str.append(r'\end{tabular}')
    latex_str.append(r'\end{adjustbox}')
    return '\n'.join(latex_str)
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
