import numpy as np
import warnings
warnings.filterwarnings("ignore")

from functions_retail import *

#Load data
path = "../data/ProductionData.csv"
df = pd.read_csv(path, sep=',',parse_dates=['date'],index_col='date')

#Choose the variables that are used in the project
final_variables = ['store_id',
                     'item_id',
                     'event_name_1',                     
                     'month',
                     'sell_price',                      
                     'wday',
                     'weekday',
                     'sales']

df = df[final_variables]

forecast = recursive_forecast(df)

print(forecast)