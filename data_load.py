import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH  = '.\\data'

available_files = os.listdir(DATA_PATH)

csv_files = [file for file in available_files if file.endswith('.csv')]


print('\033[1m' + 'Available Datasets :' + '\033[0m', '\n')

for file in csv_files:
    print(file)


suit_iofrol = pd.read_csv(os.path.join(DATA_PATH, csv_files[1]),sep=';')




