import numpy as np
import warnings
warnings.filterwarnings("ignore")

from functions_retail import *

#Load data
path = "../data/work.csv"
df = pd.read_csv(path, sep=',', parse_dates=['date'], index_col='date')

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

step1_df = cleaning_data(df)
step2_df = create_variables(step1_df)

start_training(step2_df)