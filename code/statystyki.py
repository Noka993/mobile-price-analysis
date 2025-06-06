import pandas as pd
import  os
from data import read_preprocessed_data,outliers_statistics,general_statistics,dataframe_to_latex_table
X, y = read_preprocessed_data(outliers = False)
df = pd.concat([X, y], axis=1)
print(outliers_statistics(df).to_string())
statystyki = general_statistics(df)
print(statystyki.to_string())
#print(dataframe_to_latex_table(statystyki))
#print(dataframe_to_latex_table(outliers_statistics(df).transpose()))