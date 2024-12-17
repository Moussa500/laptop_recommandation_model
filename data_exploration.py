import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\dev\ml\project\laptop_data.csv")
#Exploration
print("info",data.info())
print("description",data.describe())
print("shape",data.shape)
print(data.isnull().sum())
print(data.head())
#sns.pairplot(data)
#plt.show()
#for column in data.select_dtypes(include=np.number).columns:
#    sns.boxplot(data[column])
#    plt.title(f'Boxplot of {column}')
#    plt.show()
sns.barplot(data['Gpu'])
plt.show()
sns.barplot(data['Cpu'])
plt.show()