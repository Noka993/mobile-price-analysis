import pandas as pd
import  os
from data import read_preprocessed_data,outliers_statistics
df= read_preprocessed_data(outliers = False)
print(outliers_statistics(df).to_string())
