import pandas as pd
import numpy as np

gsd = pd.read_csv('data\gsdtsr.csv',sep=';',chunksize=20000)

data1 = next(gsd)

data2 = pd.read_csv('data\iofrol-additional-features.csv',  sep=',')


data1 = pd.concat([data1, data2['DurationGroup'].iloc[0:20000], data2['TimeGroup'].iloc[0:20000]], axis=1)