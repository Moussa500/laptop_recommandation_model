from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

data = pd.read_csv(r"C:\dev\ml\project\dataframe2.csv")

cpu_value_count = data['Cpu_brand'].value_counts().sort_values(ascending=False)
gpu_value_count = data['Gpu_brand'].value_counts().sort_values(ascending=False)
ram_value_count = data['Ram'].value_counts().sort_values(ascending=False)

gpu_value_count.plot(kind='barh', color='lightcoral')
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


label_encoder_cpu = LabelEncoder()
label_encoder_gpu = LabelEncoder()
label_encoder_company = LabelEncoder()
label_encoder_os = LabelEncoder()

data['Cpu_brand'] = label_encoder_cpu.fit_transform(data['Cpu_brand'])
data['Gpu_brand'] = label_encoder_gpu.fit_transform(data['Gpu_brand'])
data['Company'] = label_encoder_company.fit_transform(data['Company'])
data['os'] = label_encoder_os.fit_transform(data['os'])


features = ['Cpu_brand', 'Gpu_brand', 'Ram']  
for col in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

threshold = 3 
for col in features:
    z_scores = zscore(data[col])
    outliers = (abs(z_scores) > threshold)
    print(f"Nombre de valeurs aberrantes dans {col}: {outliers.sum()}")

df = pd.DataFrame(data)
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

data.drop(columns=['O', 'M'], inplace=True)

correlation_matrix = df.corr()
print("Correlation Matrix (après suppression):")
print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap (après suppression)")
plt.show()

for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        plt.figure(figsize=(8, 4))
        sns.boxplot(data[col], color='lightblue')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

        z_scores = zscore(data[col])
        outliers = (abs(z_scores) > 3)  
        print(f"Nombre de valeurs aberrantes dans {col}: {outliers.sum()}")
    else:
        print(f"Colonne catégorique {col}:")
        print(data[col].value_counts())
        print("-" * 50)
output_path = r'C:\dev\ml\project\dataframe3.csv'
data.to_csv(output_path, index=False)
