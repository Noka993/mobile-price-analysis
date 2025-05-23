import pandas as pd
import  os
from data import read_preprocessed_data,outliers_statistics,general_statistics,dataframe_to_latex_table
df_outliers = read_preprocessed_data(outliers = False)
print(outliers_statistics(df_outliers).to_string())
df = read_preprocessed_data()
statystyki = general_statistics(df)
print(statystyki.to_string())
#print(dataframe_to_latex_table(statystyki))
#print(dataframe_to_latex_table(outliers_statistics(df_outliers).transpose()))