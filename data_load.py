import os
import pandas as pd
import pandas_profiling

DATA_PATH  = './data'

available_files = os.listdir(DATA_PATH)

csv_files = [file for file in available_files if file.endswith('.csv')]

print('\033[1m' + 'Available Datasets :' + '\033[0m', '\n')

for file in csv_files:
    print(file)

commons_codec_raw = pd.read_csv(os.path.join(DATA_PATH, csv_files[0]))

print(commons_codec_raw.info())


iofrol_raw = pd.read_csv(os.path.join(DATA_PATH, csv_files[7]))

print(iofrol_raw.info())


# profile = pandas_profiling.ProfileReport(commons_codec_raw)

# profile.to_file('report_commens_codec.html')
