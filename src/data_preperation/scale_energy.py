"""small script to scale the electrical energy by 4 because in order to convert energy to power output"""
import pandas as pd

path = 'datasets/consumption/consumption.csv'
data = pd.read_csv(path, sep=';', index_col=0, parse_dates=True)
cols = [c for c in data if '[MWh]' in c]
rename_dir = {c:c.replace('[MWh]', '[MW]') for c in cols}
for c in cols:
    data[c] = data[c] * 4
    
data = data.rename(columns=rename_dir)
data.to_csv(path+'4', sep=';')

