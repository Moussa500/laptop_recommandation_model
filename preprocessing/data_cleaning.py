import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def fetch_processor(x):
  cpu_name = " ".join(x.split()[0:3])
  if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
    return cpu_name
  elif cpu_name.split()[0] == 'Intel':
    return 'Intel Dual Core'
  else:
    return 'AMD Processor'
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Linux'
data = pd.read_csv(r"C:\dev\ml\project\laptop_data.csv")
data['Touchscreen'] = data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
data['Ram'] = data['Ram'].str.replace("GB", "")
data['Ram'] = data['Ram'].astype('int32')
data['Gpu_brand'] = data['Gpu'].apply(lambda x:x.split()[0])
data['Cpu_brand'] = data['Cpu'].apply(lambda x: fetch_processor(x))
plt.figure(figsize=(10, 6))
cpu_value_count = data['Cpu_brand'].value_counts().sort_values(ascending=False)
gpu_value_count = data['Gpu_brand'].value_counts().sort_values(ascending=False)
ram_value_count = data['Ram'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
gpu_value_count.plot(kind='barh', color='lightcoral',)
plt.title('Counts of Each GPU Brand')
plt.xlabel('Count')
plt.ylabel('GPU Brand')
plt.show()
cpu_value_count.plot(kind='barh', color='lightcoral')
plt.title('Counts of Each CPU Brand')
plt.xlabel('Count')
plt.ylabel('CPU Brand')
plt.show()
ram_value_count.plot(kind='barh', color='lightcoral')
plt.title('Counts Ram')
plt.xlabel('Count')
plt.ylabel('Ram')
plt.show()
output_path = r'C:\dev\ml\dataset.csv'
data = data[data['Gpu_brand'] != 'ARM']
data = data[data['Ram'] != 64]
data = data[data['Ram'] != 24]
data['P'] = ((data['Price'] *0.04) / 1_000).round(3)
data['os'] = data['OpSys'].apply(cat_os)
data.drop(columns=['OpSys'],inplace=True)
data.drop(columns=['Unnamed: 0','Weight','TypeName', 'Memory','ScreenResolution','Inches','Gpu','Cpu','Price'], inplace=True)
output_path = r'C:\dev\ml\project\dataframe.csv'
data.to_csv(output_path, index=False)
